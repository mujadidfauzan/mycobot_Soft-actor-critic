from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class ResolvedObject:
    body_name: str
    joint_name: str
    frame_site_name: str | None
    joint_id: int
    qpos_adr: int
    dof_adr: int


def resolve_object(model: mujoco.MjModel) -> ResolvedObject:
    """
    Resolve which object body/joint the env should use.

    This repo has multiple XML variants (e.g. a single `obj` freejoint, or
    multi-object scenes with `cube_obj`, `triangle_obj`, `cylinder_obj`).
    We try a small set of known name triples and pick the first that exists.
    """

    candidates = [
        ("obj", "obj_joint", "obj_frame"),
        ("obj", "obj_joint", "obj_site"),
        ("cube_obj", "cube", "cube_frame"),
        ("triangle_obj", "triangle_joint", "triangle_frame"),
        ("cylinder_obj", "cylinder_joint", "cylinder_frame"),
    ]

    for body_name, joint_name, site_name in candidates:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if joint_id == -1 or body_id == -1:
            continue

        resolved_site_name: str | None = None
        if site_name is not None:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id != -1:
                resolved_site_name = site_name

        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        return ResolvedObject(
            body_name=body_name,
            joint_name=joint_name,
            frame_site_name=resolved_site_name,
            joint_id=joint_id,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
        )

    raise ValueError(
        "Could not resolve object in the loaded MuJoCo model. Expected one of: "
        "('obj','obj_joint'), ('cube_obj','cube'), ('triangle_obj','triangle_joint'), "
        "('cylinder_obj','cylinder_joint')."
    )


def resolve_known_objects(model: mujoco.MjModel) -> dict[str, ResolvedObject]:
    """
    Resolve all known object variants that exist in the model.

    Keys are: 'obj', 'cube', 'triangle', 'cylinder'.
    """

    specs: dict[str, tuple[str, str, str | None]] = {
        "obj": ("obj", "obj_joint", "obj_frame"),
        "cube": ("cube_obj", "cube", "cube_frame"),
        "triangle": ("triangle_obj", "triangle_joint", "triangle_frame"),
        "cylinder": ("cylinder_obj", "cylinder_joint", "cylinder_frame"),
    }

    resolved: dict[str, ResolvedObject] = {}
    for key, (body_name, joint_name, site_name) in specs.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if joint_id == -1 or body_id == -1:
            continue

        resolved_site_name: str | None = None
        if site_name is not None:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id != -1:
                resolved_site_name = site_name

        resolved[key] = ResolvedObject(
            body_name=body_name,
            joint_name=joint_name,
            frame_site_name=resolved_site_name,
            joint_id=joint_id,
            qpos_adr=int(model.jnt_qposadr[joint_id]),
            dof_adr=int(model.jnt_dofadr[joint_id]),
        )

    # Backward-compat: if we have an 'obj' joint but the site name differs.
    if "obj" in resolved and resolved["obj"].frame_site_name is None:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "obj_site")
        if site_id != -1:
            resolved["obj"] = ResolvedObject(
                body_name=resolved["obj"].body_name,
                joint_name=resolved["obj"].joint_name,
                frame_site_name="obj_site",
                joint_id=resolved["obj"].joint_id,
                qpos_adr=resolved["obj"].qpos_adr,
                dof_adr=resolved["obj"].dof_adr,
            )

    return resolved
