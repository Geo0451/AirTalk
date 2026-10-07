extends Camera3D


@export var sensitivity: float = 0.003
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# Hide the mouse cursor and lock it to the center of the screen
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _unhandled_input(event: InputEvent) -> void:

	if event is InputEventMouseMotion:
		# Rotate the whole character left and right (Y-axis)
		rotate_y(-event.relative.x * sensitivity)

		# Rotate the camera pivot up and down (X-axis)
		rotate_x(-event.relative.y * sensitivity)

		# Limit the up/down look angle so the camera does not flip over
		rotation.x = clamp(
			rotation.x, 
			deg_to_rad(-90), 
			deg_to_rad(90)
		)
	

func _input(event: InputEvent) -> void:
	# Press Escape to release the mouse cursor
	if event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
