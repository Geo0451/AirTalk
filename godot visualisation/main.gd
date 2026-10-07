extends Node3D

@export_group("Network Settings")
@export var listen_port: int = 5005

@export_group("Canvas & Sensitivity")
@export var canvas_pixel_size: Vector2 = Vector2(1024, 1024)
@export var sensitivity: Vector2 = Vector2(16.0, 16.0) ## Pixels per degree of rotation
@export var madgwick_beta: float = 0.08

@export_group("Node References")
@export var mesh_path: NodePath = "MeshInstance3D"
@export var trail_canvas_path: NodePath = "LaserCanvas/SubViewport/TrailCanvas"

var _mesh: Node3D = null
var _trail_canvas: Node = null
var udp := PacketPeerUDP.new()

# Orientation state (radians)
var pitch: float = 0.0
var yaw: float = 0.0
var roll: float = 0.0

# Anchor state for 2D positioning
var pitch_anchor: float = 0.0
var yaw_anchor: float = 0.0
var anchor_set: bool = false

var q_madgwick := Quaternion.IDENTITY
var is_drawing: bool = false
var _last_ts_us: int = 0

# Network Debug Counters
var _packets_this_second: int = 0
var _debug_timer: float = 0.0

func _ready() -> void:
	if has_node(mesh_path):
		_mesh = get_node(mesh_path) as Node3D
	if has_node(trail_canvas_path):
		_trail_canvas = get_node(trail_canvas_path)
	else:
		push_error("[Godot] Could not find TrailCanvas at path: ", trail_canvas_path)

	udp.close()
	# Bind to 0.0.0.0 (all interfaces) to catch UDP broadcast packets on port 5005
	var err := udp.bind(listen_port, "*")
	if err == OK:
		print("[UDP Debug] SUCCESS: Bound to port %d. Waiting for ESP32 packets..." % listen_port)
	else:
		push_error("[UDP Debug] FAILED to bind port %d. Error code: %d" % [listen_port, err])

func _process(delta: float) -> void:
	# --- UDP PACKET INSPECTOR ---
	_debug_timer += delta
	if _debug_timer >= 1.0:
		if _packets_this_second == 0:
			print("[UDP Debug] WARN: Received 0 packets in the last second. Check IP subnet/firewall.")
		else:
			print("[UDP Debug] Receiving OK: %d packets/sec | Touch Active: %s" % [_packets_this_second, is_drawing])
		_packets_this_second = 0
		_debug_timer = 0.0

	while udp.get_available_packet_count() > 0:
		var pkt := udp.get_packet()
		_packets_this_second += 1

		# Validate binary struct size (29 bytes expected)
		if pkt.size() != 29:
			push_warning("[UDP Debug] Unexpected packet byte length: %d (expected 29)" % pkt.size())
			continue

		# Unpack binary payload with explicit Little-Endian decoding
		var buffer := StreamPeerBuffer.new()
		buffer.big_endian = false
		buffer.data_array = pkt

		var timestamp_us: int = buffer.get_u32()
		var touched: bool = (buffer.get_u8() == 1)

		var ax: float = buffer.get_float()
		var ay: float = buffer.get_float()
		var az: float = buffer.get_float()

		var gx_deg: float = buffer.get_float()
		var gy_deg: float = buffer.get_float()
		var gz_deg: float = buffer.get_float()

		# Delta time calculation from microsecond hardware ticks
		var dt: float = 0.01
		if _last_ts_us > 0 and timestamp_us > _last_ts_us:
			dt = clampf((timestamp_us - _last_ts_us) / 1_000_000.0, 0.0005, 0.05)
		_last_ts_us = timestamp_us

		# Raw IMU vectors (Unswapped for Madgwick gradient math)
		var g_raw := Vector3(deg_to_rad(gx_deg), deg_to_rad(gy_deg), deg_to_rad(gz_deg))
		var a_raw := Vector3(ax, ay, az)

		# --- TOUCH STATE TRANSITION ---
		if touched and not is_drawing:
			is_drawing = true
			print("[AirStylus] Pen Touch DOWN")

			# Sync gyro integration state to current Madgwick baseline orientation
			var euler := q_madgwick.get_euler()
			pitch = -euler.x  # Inverted pitch polarity match
			yaw   = euler.z
			roll  = euler.y

			# Lock canvas center anchor on initial touch
			if not anchor_set:
				pitch_anchor = pitch
				yaw_anchor = yaw
				anchor_set = true
				print("[AirStylus] Anchor locked to current pen direction.")

		elif not touched and is_drawing:
			is_drawing = false
			print("[AirStylus] Pen Touch UP")
			if _trail_canvas and _trail_canvas.has_method("lift_pen"):
				_trail_canvas.lift_pen()

		# --- HYBRID ORIENTATION FILTER ---
		if is_drawing:
			# PEN DOWN: Pure Gyro Integration
			# Physical mapping: gx = Pitch, gz = Yaw, gy = Roll
			pitch -= deg_to_rad(gx_deg) * dt
			yaw   += deg_to_rad(gz_deg) * dt
			roll  += deg_to_rad(gy_deg) * dt
		else:
			# PEN UP: Madgwick Filter pull back to gravity
			q_madgwick = _update_madgwick(q_madgwick, g_raw, a_raw, dt)
			var euler := q_madgwick.get_euler()
			pitch = -euler.x
			yaw   = euler.z
			roll  = euler.y

		# Update 3D Stylus Mesh rotation for visual feedback
		if _mesh:
			_mesh.quaternion = Quaternion.from_euler(Vector3(pitch, yaw, roll))

		# --- DIRECT 2D ANGULAR CANVAS MAPPING ---
		if is_drawing and _trail_canvas:
			var delta_pitch_deg: float = rad_to_deg(pitch - pitch_anchor)
			var delta_yaw_deg: float   = rad_to_deg(yaw - yaw_anchor)

			var canvas_center := canvas_pixel_size * 0.5
			var canvas_pos := Vector2(
				canvas_center.x + (delta_yaw_deg * sensitivity.x),
				canvas_center.y - (delta_pitch_deg * sensitivity.y)
			)

			canvas_pos.x = clampf(canvas_pos.x, 0.0, canvas_pixel_size.x)
			canvas_pos.y = clampf(canvas_pos.y, 0.0, canvas_pixel_size.y)

			if _trail_canvas.has_method("add_point"):
				_trail_canvas.add_point(canvas_pos)

# Recenter the 2D cursor anchor to current pen direction
func recenter_anchor() -> void:
	pitch_anchor = pitch
	yaw_anchor = yaw
	print("[AirStylus] Recaptured center anchor.")

# --- MADGWICK AHRS IMPLEMENTATION ---
func _update_madgwick(q_in: Quaternion, g: Vector3, a: Vector3, dt: float) -> Quaternion:
	var q1: float = q_in.w; var q2: float = q_in.x; var q3: float = q_in.y; var q4: float = q_in.z
	var a_len: float = a.length()
	if a_len < 0.001:
		return q_in

	var ax: float = a.x / a_len; var ay: float = a.y / a_len; var az: float = a.z / a_len

	var _2q1: float = 2.0 * q1; var _2q2: float = 2.0 * q2; var _2q3: float = 2.0 * q3; var _2q4: float = 2.0 * q4
	var _4q1: float = 4.0 * q1; var _4q2: float = 4.0 * q2; var _4q3: float = 4.0 * q3
	var _8q2: float = 8.0 * q2; var _8q3: float = 8.0 * q3
	var q1q1: float = q1 * q1; var q2q2: float = q2 * q2; var q3q3: float = q3 * q3; var q4q4: float = q4 * q4

	var s1: float = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
	var s2: float = _4q2 * q4q4 - _2q4 * ax + 4.0 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
	var s3: float = 4.0 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
	var s4: float = 4.0 * q2q2 * q4 - _2q2 * ax + 4.0 * q3q3 * q4 - _2q3 * ay

	var s_len: float = sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4)
	if s_len > 0.00001:
		s1 /= s_len; s2 /= s_len; s3 /= s_len; s4 /= s_len

	var qDot1: float = 0.5 * (-q2 * g.x - q3 * g.y - q4 * g.z) - madgwick_beta * s1
	var qDot2: float = 0.5 * ( q1 * g.x + q3 * g.z - q4 * g.y) - madgwick_beta * s2
	var qDot3: float = 0.5 * ( q1 * g.y - q2 * g.z + q4 * g.x) - madgwick_beta * s3
	var qDot4: float = 0.5 * ( q1 * g.z + q2 * g.y - q3 * g.x) - madgwick_beta * s4

	q1 += qDot1 * dt
	q2 += qDot2 * dt
	q3 += qDot3 * dt
	q4 += qDot4 * dt

	return Quaternion(q2, q3, q4, q1).normalized()
