extends Node3D
## LASER POINTER
## ---------------------------------------------------------------------
## Attach this as a CHILD of the pen's MeshInstance3D (or any node whose
## transform tracks the stylus tip). It inherits the stylus orientation
## automatically, so no wiring back to main.gd is needed.
##
## Every physics frame it:
##   1. Casts a ray forward from its own position.
##   2. Draws a visible emissive "laser" mesh from the tip to the hit point
##      (or to max range, if nothing was hit).
##   3. If it hits something in the "laser_canvas" group, calls that
##      object's laser_hit(world_pos, normal) so it can paint a trail.
##   4. If it *stops* hitting the canvas (pen lifted / moved off edge),
##      calls laser_lift() so the next stroke doesn't connect to this one.

@export_group("Laser")
@export var laser_length: float = 5.0
@export var laser_color: Color = Color(1.0, 0.15, 0.15)
@export var laser_width: float = 0.004
@export var collision_mask: int = 1

@export_group("Aim")
## Direction the laser points, in THIS node's local space.
## Change this if your pen model's "forward" isn't -Z.
@export var local_forward: Vector3 = Vector3.FORWARD

var _beam: MeshInstance3D
var _was_hitting_canvas: bool = false


func _ready() -> void:
	_build_beam()


func _build_beam() -> void:
	var mesh := CylinderMesh.new()
	mesh.top_radius = laser_width
	mesh.bottom_radius = laser_width
	mesh.height = 1.0
	mesh.radial_segments = 8

	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = laser_color
	mat.emission_enabled = true
	mat.emission = laser_color
	mat.emission_energy_multiplier = 4.0
	mesh.material = mat

	_beam = MeshInstance3D.new()
	_beam.mesh = mesh
	_beam.visible = false
	# Parented to the scene root (not to this node) so the stylus's own
	# scale/rotation never distorts the beam geometry.
	get_tree().current_scene.add_child.call_deferred(_beam)


func _physics_process(_delta: float) -> void:
	if not is_instance_valid(_beam):
		return

	var origin: Vector3 = global_transform.origin
	var dir: Vector3 = (global_transform.basis * local_forward).normalized()
	var to: Vector3 = origin + dir * laser_length

	var space_state := get_world_3d().direct_space_state
	var query := PhysicsRayQueryParameters3D.create(origin, to)
	query.collision_mask = collision_mask
	var result := space_state.intersect_ray(query)

	var end_point: Vector3 = to
	var hit_canvas := false

	if result:
		end_point = result.position
		var collider = result.collider
		if collider and collider.is_in_group("laser_canvas") and collider.has_method("laser_hit"):
			collider.laser_hit(result.position, result.normal)
			hit_canvas = true

	if _was_hitting_canvas and not hit_canvas:
		for c in get_tree().get_nodes_in_group("laser_canvas"):
			if c.has_method("laser_lift"):
				c.laser_lift()
	_was_hitting_canvas = hit_canvas

	_update_beam(origin, end_point)


func _update_beam(from: Vector3, to: Vector3) -> void:
	var dist := from.distance_to(to)
	if dist < 0.001:
		_beam.visible = false
		return

	var dir_n := (to - from) / dist
	var up := Vector3.UP
	if absf(dir_n.dot(up)) > 0.99:
		up = Vector3.RIGHT

	# CylinderMesh's height runs along local +Y, so align the basis Y
	# column with the beam direction.
	var basis := Basis()
	basis.y = dir_n
	basis.x = dir_n.cross(up).normalized()
	basis.z = basis.x.cross(dir_n).normalized()

	var mid := (from + to) * 0.5
	_beam.global_transform = Transform3D(basis, mid)
	_beam.scale = Vector3(1.0, dist, 1.0)
	_beam.visible = true
