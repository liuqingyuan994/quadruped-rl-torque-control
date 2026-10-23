"""Reproducible conversion of the supplied Mini Lite URDF and DAE assets."""
import hashlib
import json
import math
import re
import struct
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets'
LEGS = ('FL', 'FR', 'RL', 'RR')
SDK_LEGS = ('FL', 'FR', 'HL', 'HR')
MODEL_ID = 'mini_lite_urdf_20261019_v1'
SOURCE = ASSETS / 'mini_lite_source'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_policy_compatible(configuration):
    if configuration.get('robot_model') != MODEL_ID:
        raise ValueError('策略模型不匹配：当前为用户提供的 Mini Lite URDF。请使用本模型的策略或重新训练。')
    dt = configuration.get('physics_dt', 0) * configuration.get('frame_skip', 0)
    if abs(dt - .002) > 1e-12:
        raise ValueError('Mini Lite 控制周期必须为 2 ms。')


def numbers(text):
    return np.fromstring(text, sep=' ')


def dae_to_stl(source, target):
    """Convert the supplied metre/Z_UP triangle scenes without mesh rescaling.

    This intentionally accepts only the constructs in this source package;
    unsupported transforms/primitives fail instead of silently changing geometry.
    """
    root = ET.parse(source).getroot()
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    if root.findtext('c:asset/c:up_axis', namespaces=ns) != 'Z_UP':
        raise ValueError('Expected source DAE in Z_UP coordinates')
    unit = float(root.find('c:asset/c:unit', ns).get('meter'))
    if unit != 1:
        raise ValueError('Expected source mesh in metres')
    scene_id = root.find('c:scene/c:instance_visual_scene', ns).get('url')[1:]
    scene = root.find(f"c:library_visual_scenes/c:visual_scene[@id='{scene_id}']", ns)
    refs = []
    def visit(node):
        if node.findall('.//c:instance_geometry', ns):
            for child in node:
                tag = child.tag.split('}')[-1]
                if tag in ('translate', 'rotate', 'scale', 'matrix', 'lookat', 'skew'):
                    vals = numbers(child.text)
                    identity = (tag == 'matrix' and np.allclose(vals.reshape(4, 4), np.eye(4)))
                    if not identity:
                        raise ValueError(f'Unexpected geometry transform in {source}: {tag}')
        refs.extend(x.get('url')[1:] for x in node.findall('c:instance_geometry', ns))
        for child in node.findall('c:node', ns):
            visit(child)
    for node in scene.findall('c:node', ns):
        visit(node)
    triangles = []
    for ref in refs:
        mesh = root.find(f"c:library_geometries/c:geometry[@id='{ref}']/c:mesh", ns)
        sources = {}
        for src in mesh.findall('c:source', ns):
            accessor = src.find('c:technique_common/c:accessor', ns)
            stride = int(accessor.get('stride', '1'))
            offset = int(accessor.get('offset', '0'))
            count = int(accessor.get('count'))
            arr = src.find(f"c:float_array[@id='{accessor.get('source')[1:]}']", ns)
            sources[src.get('id')] = numbers(arr.text)[offset:offset+count*stride].reshape(count, stride)
        vertices = {v.get('id'): next(i.get('source')[1:] for i in v.findall('c:input', ns) if i.get('semantic') == 'POSITION')
                    for v in mesh.findall('c:vertices', ns)}
        for primitive in mesh:
            tag = primitive.tag.split('}')[-1]
            if tag in ('source', 'vertices'):
                continue
            if tag != 'triangles':
                raise ValueError(f'Unsupported source primitive: {tag}')
            inputs = primitive.findall('c:input', ns)
            stride = max(int(i.get('offset', '0')) for i in inputs) + 1
            vertex = next(i for i in inputs if i.get('semantic') == 'VERTEX')
            ix = np.fromstring(primitive.findtext('c:p', namespaces=ns), sep=' ', dtype=np.int64).reshape(-1, stride)
            pos = sources[vertices[vertex.get('source')[1:]]][:, :3]
            tris = pos[ix[:, int(vertex.get('offset', '0'))]].reshape(-1, 3, 3)
            if len(tris) != int(primitive.get('count')):
                raise ValueError('Triangle count mismatch')
            triangles.append(tris)
    tris = np.concatenate(triangles)
    normals = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > 1e-16
    tris, normals, norms = tris[keep], normals[keep], norms[keep]
    normals /= norms[:, None]
    records = np.zeros(len(tris), dtype=[('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    records['normal'], records['vertices'] = normals, tris
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'Mini Lite source DAE triangles; metres; no rescaling'.ljust(80, b'\0') + struct.pack('<I', len(tris)) + records.tobytes())
    return {'source': str(source.relative_to(ASSETS)), 'source_sha256': sha256(source),
            'output': str(target.relative_to(ASSETS)), 'output_sha256': sha256(target),
            'triangles': len(tris), 'degenerate_triangles_removed': int((~keep).sum()),
            'bounds_m': [tris.min(axis=(0, 1)).tolist(), tris.max(axis=(0, 1)).tolist()],
            'materials': 'Single display colour; geometry and scale retained'}


def build_model(output=None):
    original = SOURCE / 'mini_lite/urdf/MiniLiteC.urdf'
    tree = ET.parse(original)
    urdf = tree.getroot()
    links = {l.get('name'): l for l in urdf.findall('link')}
    # The supplied WBT explicitly gives the corrected HR_HIP tensor. Its other
    # five entries match the URDF; only izz was accidentally copied from iyz.
    tensor = links['HR_HIP'].find('inertial/inertia')
    bad_izz = tensor.get('izz')
    world = SOURCE / 'mini_lite/webots/worlds/mark_0_init_model.wbt'
    expected = r'inertiaMatrix\s*\[\s*0\.0001497\s+0\.00026266\s+0\.00014559\s+1\.1564e-07\s+3\.5848e-06\s+-4\.1425e-07\s*\]'
    if not re.search(expected, world.read_text(encoding='utf-8'), re.I):
        raise ValueError('Source WBT no longer supports the HR_HIP inertia correction')
    if float(bad_izz) != -4.1425e-7:
        raise ValueError('Unexpected source inertia; review the correction')
    tensor.set('izz', '0.00014559')
    inertias = []
    for name, link in links.items():
        inertial = link.find('inertial')
        if inertial is None:
            continue
        a = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
        matrix = np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]])
        eigen = np.linalg.eigvalsh(matrix)
        if min(eigen) <= 0 or eigen[0]+eigen[1] < eigen[2]-1e-14:
            raise ValueError(f'Invalid physical inertia: {name}')
        inertias.append({'link': name, 'mass_kg': float(inertial.find('mass').get('value')), 'eigenvalues_kg_m2': eigen.tolist()})
    runtime_urdf = SOURCE / 'mini_lite/urdf/MiniLiteC.runtime.urdf'
    ET.indent(tree)
    tree.write(runtime_urdf, encoding='utf-8', xml_declaration=True)
    conversions = [dae_to_stl(p, ASSETS/'mini_lite_meshes'/(p.stem+'.stl'))
                   for p in sorted((SOURCE/'mini_lite/meshes').glob('*.dae'))]
    root = ET.Element('mujoco', model=MODEL_ID)
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', inertiafromgeom='false', fusestatic='false')
    ET.SubElement(root, 'option', timestep='.002', gravity='0 0 -9.81', integrator='implicitfast', cone='elliptic', iterations='40')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='900', offheight='600')
    defaults = ET.SubElement(root, 'default')
    ET.SubElement(defaults, 'geom', friction='0.8 0.005 0.0001', solref='0.01 1', solimp='0.9 0.95 0.001', condim='3')
    assets = ET.SubElement(root, 'asset')
    ET.SubElement(assets, 'texture', name='checker', type='2d', builtin='checker', rgb1='.18 .22 .25', rgb2='.28 .32 .35', width='512', height='512')
    ET.SubElement(assets, 'material', name='floor', texture='checker', texrepeat='25 25', texuniform='true')
    for p in conversions:
        ET.SubElement(assets, 'mesh', name=Path(p['output']).stem, file=p['output'].replace('\\', '/'))
    worldbody = ET.SubElement(root, 'worldbody')
    ET.SubElement(worldbody, 'light', pos='0 0 4', dir='0 0 -1', directional='true')
    ET.SubElement(worldbody, 'geom', name='ground', type='plane', size='30 30 .1', material='floor')
    torso = ET.SubElement(worldbody, 'body', name='torso', pos='0 0 .31')
    ET.SubElement(torso, 'freejoint', name='root')
    def origin(element):
        node = element.find('origin')
        return {'pos': node.get('xyz', '0 0 0'), 'euler': node.get('rpy', '0 0 0')} if node is not None else {}
    def inertia(body, link):
        node = link.find('inertial')
        if node is not None:
            import mujoco
            a = {k: float(v) for k, v in node.find('inertia').attrib.items()}
            if 'euler' in origin(node) and origin(node)['euler'] != '0 0 0':
                raise ValueError('Nonzero inertia RPY needs explicit tensor rotation')
            matrix = np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]])
            diagonal, axes = np.linalg.eigh(matrix)
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(axes).ravel())
            # Explicit principal axes avoid the compiler's approximate Jacobi
            # eigensolver rounding small off-diagonal source inertia terms.
            ET.SubElement(body, 'inertial', pos=origin(node).get('pos', '0 0 0'), mass=node.find('mass').get('value'),
                          diaginertia=' '.join(map(str, diagonal)), quat=' '.join(map(str, quat)))
    def geometry(body, link, prefix):
        for k, element in enumerate(link.findall('visual')):
            mesh = element.find('geometry/mesh')
            if mesh is None or mesh.get('scale', '1 1 1') != '1 1 1':
                raise ValueError('Unexpected source visual')
            ET.SubElement(body, 'geom', name=prefix+'_visual'+str(k), type='mesh', mesh=Path(mesh.get('filename')).stem,
                          **origin(element), mass='0', contype='0', conaffinity='0', rgba='.60 .63 .67 1', group='2')
        for k, element in enumerate(link.findall('collision')):
            shape = list(element.find('geometry'))[0]
            attrs = {'name': prefix.removesuffix('_foot_body')+'_foot_geom' if prefix.endswith('_foot_body') else prefix+'_collision'+str(k),
                     **origin(element), 'mass': '0', 'rgba': '0 0 0 0'}
            if shape.tag == 'box':
                attrs.update(type='box', size=' '.join(str(v/2) for v in numbers(shape.get('size'))))
            elif shape.tag == 'sphere':
                attrs.update(type='sphere', size=shape.get('radius'), rgba='.10 .12 .14 1', priority='1')
            else:
                raise ValueError('Unsupported source collision')
            ET.SubElement(body, 'geom', **attrs)
    inertia(torso, links['BASE_INER'])
    geometry(torso, links['BASE'], 'torso')
    ET.SubElement(torso, 'site', name='imu', size='.008', pos='0 0 0')
    joint_map = []
    for leg, sdk in zip(LEGS, SDK_LEGS):
        parent = torso
        for part, source_joint, source_link, internal_joint in (
                ('hip', 'HipX', 'HIP', 'abduction'), ('thigh', 'HipY', 'THIGH', 'hip_flexion'),
                ('shank', 'Knee', 'SHANK', 'knee'), ('foot_body', 'Ankle', 'FOOT', None)):
            joint = urdf.find(f"joint[@name='{sdk}_{source_joint}']")
            link = links[f'{sdk}_{source_link}']
            body = ET.SubElement(parent, 'body', name=f'{leg}_{part}', **origin(joint))
            if internal_joint:
                limit = joint.find('limit')
                dyn = joint.find('dynamics')
                dyn = dyn.attrib if dyn is not None else {}
                ET.SubElement(body, 'joint', name=f'{leg}_{internal_joint}', axis=joint.find('axis').get('xyz'),
                              range=limit.get('lower')+' '+limit.get('upper'), damping=dyn.get('damping', '0'), frictionloss=dyn.get('friction', '0'), armature='0')
                joint_map.append({'index': len(joint_map), 'sdk_joint': joint.get('name'), 'simulation_joint': f'{leg}_{internal_joint}',
                                  'axis': numbers(joint.find('axis').get('xyz')).tolist(), 'range_rad': [float(limit.get('lower')), float(limit.get('upper'))],
                                  'effort_Nm': float(limit.get('effort')), 'velocity_rad_s': float(limit.get('velocity'))})
            inertia(body, link)
            geometry(body, link, f'{leg}_{part}')
            if part == 'foot_body':
                ET.SubElement(body, 'site', name=leg+'_foot', size='.004', rgba='.2 .7 .4 0')
            parent = body
    actuator = ET.SubElement(root, 'actuator')
    for joint in joint_map:
        limit = joint['effort_Nm']
        name = joint['simulation_joint']
        ET.SubElement(actuator, 'motor', name=name+'_motor', joint=name, ctrlrange=f'{-limit} {limit}', gear='1')
    sensor = ET.SubElement(root, 'sensor')
    ET.SubElement(sensor, 'gyro', name='gyro', site='imu')
    ET.SubElement(sensor, 'accelerometer', name='accelerometer', site='imu')
    ET.SubElement(sensor, 'framequat', name='orientation', objtype='site', objname='imu')
    keys = ET.SubElement(root, 'keyframe')
    qpos = [0, 0, .31, 1, 0, 0, 0] + [0, -.65, 1.3]*4
    ET.SubElement(keys, 'key', name='stand', qpos=' '.join(map(str, qpos)))
    ET.indent(root)
    path = Path(output) if output else ASSETS/'quadruped.xml'
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    manifest = {'model_id': MODEL_ID, 'source_urdf_sha256': sha256(original), 'runtime_urdf_sha256': sha256(runtime_urdf),
                'mjcf_sha256': sha256(path), 'total_mass_kg': sum(i['mass_kg'] for i in inertias), 'torso_mass_kg': 5.6352,
                'policy_action_scale': .4, 'control_period_s': .002, 'joints': joint_map,
                'inertias': inertias, 'mesh_conversions': conversions,
                'corrections': [{'link': 'HR_HIP', 'field': 'inertial/inertia/izz', 'original': float(bad_izz), 'runtime': .00014559,
                                 'evidence': str(world.relative_to(ASSETS)), 'evidence_sha256': sha256(world), 'evidence_lines': [744, 755]}],
                'source_precedence': 'URDF dynamics, transforms, axes, joint limits, effort, damping and collisions; SDK manual control period',
                'manual_conflicts': {'total_mass_kg': {'urdf': 10.009390004, 'manual_old_project': 12.98},
                                     'motor_torque_limit_Nm': {'urdf': [13.2, 13.2, 28], 'manual_peak': [26, 26, 51]}},
                'simulation_assumptions': {'gravity_m_s2': 9.81, 'contact_friction': [.8, .005, .0001], 'contact_solref': [.01, 1],
                                           'initial_height_m': .31, 'neutral_angles_rad': [0, -.65, 1.3],
                                           'fixed_BASE_INER': 'Aggregated into torso at the identical frame',
                                           'collision_geometry': 'Exact supplied base/thigh boxes and foot spheres; no added hip/shank collisions',
                                           'imu': 'Synthetic site at BASE origin; physical extrinsics not supplied'}}
    (ASSETS/'robot_parameters.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build_model())
