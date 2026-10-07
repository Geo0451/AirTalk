extends StaticBody3D

@export var canvas_world_size: Vector2 = Vector2(2.0, 2.0)
@export var canvas_pixel_size: Vector2i = Vector2i(1024, 1024)

@onready var _mesh: MeshInstance3D = $MeshInstance3D
@onready var _viewport: SubViewport = $SubViewport
@onready var _trail = $SubViewport/TrailCanvas


func _ready() -> void:
	add_to_group("laser_canvas")

	# Force SubViewport to always render, not just "when visible"
	_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	_viewport.transparent_bg = true

	# Must wait one frame so the viewport texture is initialised
	await get_tree().process_frame

	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_texture = _viewport.get_texture()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	_mesh.material_override = mat

	# Make the canvas itself visible as a dark board in the world
	var bg_mat := StandardMaterial3D.new()
	bg_mat.albedo_color = Color(0.05, 0.05, 0.1)
	# We can't use the same mesh for bg, so just tint via albedo modulate
	# Instead, set a background color in TrailCanvas's _draw


func laser_hit(world_pos: Vector3, _normal: Vector3) -> void:
	var local_pos: Vector3 = to_local(world_pos)
	var uv := Vector2(
		(local_pos.x / canvas_world_size.x) + 0.5,
		0.5 - (local_pos.y / canvas_world_size.y)
	)
	if uv.x < 0.0 or uv.x > 1.0 or uv.y < 0.0 or uv.y > 1.0:
		laser_lift()
		return
	_trail.add_point(uv * Vector2(canvas_pixel_size))


func laser_lift() -> void:
	_trail.lift_pen()
