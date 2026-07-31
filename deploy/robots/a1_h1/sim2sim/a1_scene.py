"""Build a MuJoCo table-tennis scene from the IsaacLab A1 training URDF.

The real robot path in ``h1_pingpong`` drives DAMIAO motors with MIT commands.
For sim2sim we keep the same control abstraction and apply equivalent joint
torques directly in MuJoCo instead of going through Unitree DDS.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def guess_lgy_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Pingpong_TTRL").exists() and (parent / "unitree_rl_lab").exists():
            return parent
    return Path(__file__).resolve().parents[4]


LGY_ROOT = guess_lgy_root()
PINGPONG_TTRL = LGY_ROOT / "Pingpong_TTRL"
A1_ASSET_ROOT = PINGPONG_TTRL / "legged_lab/assets/a1/X1_URDF_V1_3"
DEFAULT_URDF = A1_ASSET_ROOT / "urdf/X1_URDF_V1_3.urdf"
DEFAULT_MESHDIR = A1_ASSET_ROOT / "meshes"
DEFAULT_SCENE_XML = Path(__file__).resolve().parent / "scene/a1_tt_scene.xml"

ROBOT_TABLE_POS = np.array([-2.0, 0.76, 0.0282], dtype=np.float64)
BALL_RADIUS = 0.02
TABLE_HEIGHT = 0.76
PHYSICS_DT = 0.002


RIGHT_ARM_JOINTS = [f"r{i}" for i in range(1, 8)]

# DAMIAO command envelope from h1_pingpong/src/armcontrol:
# h1_pingpong/src/armcontrol/src/inference_arm_control_node.cpp initializes the
# A1 right arm as r1-r3 DM4340_48V and r4-r7 DM4310_48V. These ranges are not
# a complete motor dynamics model; they are the same actuator-force metadata
# Unitree keeps in MJCF instead of URDF.
RIGHT_ARM_EFFORT_LIMITS = {
    "r1": 28.0,
    "r2": 28.0,
    "r3": 28.0,
    "r4": 10.0,
    "r5": 10.0,
    "r6": 10.0,
    "r7": 10.0,
}
RIGHT_ARM_ARMATURE = {
    "r1": 0.032,
    "r2": 0.032,
    "r3": 0.032,
    "r4": 0.0018,
    "r5": 0.0018,
    "r6": 0.0018,
    "r7": 0.0018,
}
DEFAULT_JOINT_ARMATURE = 0.001
DEFAULT_JOINT_DAMPING = 0.02
DEFAULT_JOINT_FRICTIONLOSS = 0.0
RIGHT_ARM_DAMPING = 0.02
RIGHT_ARM_FRICTIONLOSS = 0.0

DEFAULT_QPOS = {
    # v13 training matches the measured real r1 centerline height of 1.15 m.
    # X1_URDF_V1_1 gives sj ~= 1.15 - (0.0282 + 1.2107 + 0.025) = -0.1139.
    "sj": -0.1139,
    "r1": -0.505,
    "r2": -1.13,
    "r3": 1.13,
    "r4": 1.02,
    "r5": -0.7,
    "r6": 0.0,
    "r7": -1.3,
    "l1": 0.0,
    "l2": 0.0,
    "l3": 0.0,
    "l4": 0.0,
    "l5": 0.0,
    "l6": 0.0,
    "l7": 0.0,
    "t01": 0.0,
    "t02": 0.0,
    "lun_r": 0.0,
    "lun_l": 0.0,
    "wxl_1_2": 0.0,
    "wxl_1_1": 0.0,
    "wxl_2_2": 0.0,
    "wxl_2_1": 0.0,
    "wxl_3_2": 0.0,
    "wxl_3_1": 0.0,
    "wxl_4_2": 0.0,
    "wxl_4_1": 0.0,
}


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _vec(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return " ".join(_fmt(float(v)) for v in values)


def _insert_after(root: ET.Element, after_tag: str, element: ET.Element) -> None:
    children = list(root)
    for idx, child in enumerate(children):
        if child.tag == after_tag:
            root.insert(idx + 1, element)
            return
    root.append(element)


def _remove_children(root: ET.Element, tag: str) -> None:
    for child in list(root):
        if child.tag == tag:
            root.remove(child)


def _setdefault_attr(element: ET.Element, key: str, value: str) -> None:
    if element.get(key) is None:
        element.set(key, value)


def _joint_elements(root: ET.Element) -> dict[str, ET.Element]:
    return {joint.get("name", ""): joint for joint in root.iter("joint") if joint.get("name")}


def _prepare_compiler(root: ET.Element, meshdir: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("meshdir", str(meshdir))


def _prepare_option(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        _insert_after(root, "compiler", option)
    option.set("timestep", _fmt(PHYSICS_DT))
    option.set("iterations", "50")
    option.set("solver", "Newton")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")


def _prepare_visual(root: ET.Element) -> None:
    _remove_children(root, "visual")
    visual = ET.Element("visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.28 0.30 0.32",
            "diffuse": "0.75 0.75 0.72",
            "specular": "0.18 0.18 0.18",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.12 0.16 0.20 1"})
    _insert_after(root, "option", visual)


def _prepare_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        _insert_after(root, "visual", asset)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.55 0.70 0.90",
            "rgb2": "0.035 0.045 0.060",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "floor_checker",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.30 0.34 0.32",
            "rgb2": "0.21 0.24 0.23",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "floor_grid",
            "texture": "floor_checker",
            "texrepeat": "10 10",
            "reflectance": "0.08",
        },
    )


def _wrap_robot(worldbody: ET.Element) -> ET.Element:
    existing = worldbody.find("./body[@name='a1_root']")
    if existing is not None:
        return existing
    robot_root = ET.Element("body", {"name": "a1_root", "pos": _vec(ROBOT_TABLE_POS)})
    for child in list(worldbody):
        worldbody.remove(child)
        robot_root.append(child)
    worldbody.insert(0, robot_root)
    return robot_root


def _decorate_joints(root: ET.Element) -> None:
    joints = _joint_elements(root)
    for name, joint in joints.items():
        if joint.get("type") == "free":
            continue
        _setdefault_attr(joint, "armature", _fmt(DEFAULT_JOINT_ARMATURE))
        _setdefault_attr(joint, "damping", _fmt(DEFAULT_JOINT_DAMPING))
        _setdefault_attr(joint, "frictionloss", _fmt(DEFAULT_JOINT_FRICTIONLOSS))

    for name in RIGHT_ARM_JOINTS:
        joint = joints.get(name)
        if joint is None:
            raise KeyError(f"right-arm joint missing from generated MJCF: {name}")
        effort = RIGHT_ARM_EFFORT_LIMITS[name]
        joint.set("actuatorfrcrange", f"{_fmt(-effort)} {_fmt(effort)}")
        joint.set("armature", _fmt(RIGHT_ARM_ARMATURE[name]))
        joint.set("damping", _fmt(RIGHT_ARM_DAMPING))
        joint.set("frictionloss", _fmt(RIGHT_ARM_FRICTIONLOSS))


def _decorate_paddle_contact(root: ET.Element) -> None:
    for geom in root.iter("geom"):
        if geom.get("mesh") != "x1_paddle":
            continue
        geom.set("name", "paddle_blade")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("friction", "0.6 0.005 0.0001")
        geom.set("solref", "0.002 1")
        geom.set("solimp", "0.95 0.99 0.001")
        return
    raise KeyError("x1_paddle geom missing from generated MJCF")


def _append_table_tennis_scene(worldbody: ET.Element) -> None:
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key_light",
            "pos": "-1.8 0.5 3.2",
            "dir": "0.35 -0.15 -1",
            "directional": "true",
            "diffuse": "0.8 0.8 0.75",
            "specular": "0.25 0.25 0.25",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "fill_light",
            "pos": "1.6 -1.6 2.4",
            "dir": "-0.45 0.35 -1",
            "directional": "true",
            "diffuse": "0.35 0.40 0.45",
            "specular": "0.08 0.08 0.08",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": "4.0 3.0 0.05",
            "material": "floor_grid",
            "contype": "1",
            "conaffinity": "1",
            "friction": "0.8 0.005 0.0001",
        },
    )
    table = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0 0 0"})
    ET.SubElement(
        table,
        "geom",
        {
            "name": "table_top",
            "type": "box",
            "pos": f"0 0 {_fmt(TABLE_HEIGHT - 0.005)}",
            "size": "1.37 0.7625 0.005",
            "rgba": "0.1 0.3 0.6 1",
            "contype": "1",
            "conaffinity": "1",
            "friction": "0.1 0.005 0.0001",
            "solref": "0.002 1",
            "solimp": "0.95 0.99 0.001",
        },
    )
    ET.SubElement(
        table,
        "geom",
        {
            "name": "table_support",
            "type": "box",
            "pos": "0 0 0.375",
            "size": "1.30 0.70 0.375",
            "rgba": "0.18 0.18 0.18 1",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "net",
            "type": "box",
            "pos": "0 0 0.8375",
            "size": "0.01 0.7625 0.0775",
            "rgba": "0.9 0.9 0.9 0.5",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ball = ET.SubElement(worldbody, "body", {"name": "ball", "pos": "1.35 0 1.03"})
    ET.SubElement(ball, "freejoint", {"name": "ball_free"})
    ET.SubElement(
        ball,
        "geom",
        {
            "name": "ball_geom",
            "type": "sphere",
            "size": _fmt(BALL_RADIUS),
            "mass": "0.0034",
            "rgba": "1 0.12 0.05 1",
            "contype": "1",
            "conaffinity": "1",
            "friction": "0.1 0.005 0.0001",
            "solref": "0.002 1",
            "solimp": "0.95 0.99 0.001",
        },
    )


def _add_actuators(root: ET.Element) -> None:
    _remove_children(root, "actuator")
    actuator = ET.Element("actuator")
    for name in RIGHT_ARM_JOINTS:
        effort = RIGHT_ARM_EFFORT_LIMITS[name]
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"{name}_motor",
                "joint": name,
                "ctrllimited": "true",
                "ctrlrange": f"{_fmt(-effort)} {_fmt(effort)}",
            },
        )
    _insert_after(root, "worldbody", actuator)


def _add_contact_pairs(root: ET.Element) -> None:
    _remove_children(root, "contact")
    contact = ET.Element("contact")
    ET.SubElement(
        contact,
        "pair",
        {
            "geom1": "ball_geom",
            "geom2": "table_top",
            "solref": "-490000 -2",
            "solimp": "0.95 0.99 0.001",
            "margin": "0.06",
        },
    )
    ET.SubElement(
        contact,
        "pair",
        {
            "geom1": "ball_geom",
            "geom2": "paddle_blade",
            "solref": "-200000 -5",
            "solimp": "0.95 0.99 0.001",
            "margin": "0.06",
        },
    )
    _insert_after(root, "actuator", contact)


def _add_sensors(root: ET.Element) -> None:
    _remove_children(root, "sensor")
    sensor = ET.Element("sensor")
    for name in RIGHT_ARM_JOINTS:
        ET.SubElement(sensor, "jointpos", {"name": f"{name}_pos", "joint": name})
    for name in RIGHT_ARM_JOINTS:
        ET.SubElement(sensor, "jointvel", {"name": f"{name}_vel", "joint": name})
    for name in RIGHT_ARM_JOINTS:
        ET.SubElement(sensor, "jointactuatorfrc", {"name": f"{name}_torque", "joint": name})
    _insert_after(root, "contact", sensor)


def build_scene_xml(
    urdf_path: Path | str = DEFAULT_URDF,
    meshdir: Path | str = DEFAULT_MESHDIR,
    out_path: Path | str | None = None,
) -> Path:
    """Build a reusable MJCF scene from the training URDF.

    The URDF carries geometry, joint axes, ranges, masses, and inertias. This
    MJCF adds MuJoCo-only deployment metadata: passive joint dynamics, DAMIAO
    actuator force ranges, motor actuators, sensors, and table-tennis contact
    pairs.
    """

    urdf_path = Path(urdf_path)
    meshdir = Path(meshdir)
    if out_path is None:
        out_path = DEFAULT_SCENE_XML
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="a1_mjcf_") as tmpdir:
        base_xml = Path(tmpdir) / "base.xml"
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        mujoco.mj_saveLastXML(str(base_xml), model)
        tree = ET.parse(base_xml)

    root = tree.getroot()
    root.set("model", "a1_tt_scene")
    _prepare_compiler(root, meshdir)
    _prepare_option(root)
    _prepare_visual(root)
    _prepare_assets(root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("generated MJCF has no worldbody")
    _wrap_robot(worldbody)
    _decorate_joints(root)
    _decorate_paddle_contact(root)
    _append_table_tennis_scene(worldbody)
    _add_actuators(root)
    _add_contact_pairs(root)
    _add_sensors(root)

    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", short_empty_elements=True)
    out_path.write_text(out_path.read_text() + "\n")
    return out_path


def load_scene(scene_xml: Path | str | None = DEFAULT_SCENE_XML) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    if scene_xml is None:
        xml_path = build_scene_xml(out_path=Path(tempfile.gettempdir()) / "a1_h1_tt_scene.xml")
    else:
        xml_path = Path(scene_xml)
        if not xml_path.exists():
            xml_path = build_scene_xml(out_path=xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = PHYSICS_DT
    _stabilize_robot_dofs(model)
    data = mujoco.MjData(model)
    reset_robot_qpos(model, data)
    mujoco.mj_forward(model, data)
    return model, data, xml_path


def _stabilize_robot_dofs(model: mujoco.MjModel) -> None:
    """Add the actuator armature used during IsaacLab training.

    The URDF itself has no MuJoCo actuators, so without this the explicit
    torque loop is unrealistically twitchy. Leave the ball free joint untouched.
    """

    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof = int(model.jnt_dofadr[jid])
        model.dof_armature[dof] = max(float(model.dof_armature[dof]), 0.001)
        model.dof_damping[dof] = max(float(model.dof_damping[dof]), 0.02)
    for idx, name in enumerate(RIGHT_ARM_JOINTS):
        try:
            dof = joint_dof_addr(model, name)
        except KeyError:
            continue
        model.dof_armature[dof] = 0.032 if idx < 3 else 0.0018


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"joint not found: {joint_name}")
    return int(model.jnt_qposadr[jid])


def joint_dof_addr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"joint not found: {joint_name}")
    return int(model.jnt_dofadr[jid])


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"body not found: {body_name}")
    return int(bid)


def reset_robot_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name, value in DEFAULT_QPOS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
            data.qvel[model.jnt_dofadr[jid]] = 0.0


def ball_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    if jid < 0:
        raise KeyError("ball_free joint not found")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def set_ball_state(model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray, vel: np.ndarray) -> None:
    qadr, vadr = ball_addresses(model)
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[vadr : vadr + 3] = vel
    data.qvel[vadr + 3 : vadr + 6] = 0.0
