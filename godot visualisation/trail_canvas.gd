extends Node2D

@export var trail_lifetime: float = 3.0
@export var stroke_color: Color = Color(1.0, 0.2, 0.15)
@export var stroke_width: float = 6.0
@export var bg_color: Color = Color(0.05, 0.05, 0.12)

var _points: Array = []
var _pending_new_stroke: bool = true


func _ready() -> void:
	queue_redraw()


func _process(_delta: float) -> void:
	var now := Time.get_ticks_msec() / 1000.0
	var dropped := false
	while _points.size() > 0 and now - _points[0].t > trail_lifetime:
		_points.pop_front()
		dropped = true
	if dropped or _points.size() > 0:
		queue_redraw()


func add_point(pos: Vector2) -> void:
	_points.append({
		"pos": pos,
		"t": Time.get_ticks_msec() / 1000.0,
		"new_stroke": _pending_new_stroke,
	})
	_pending_new_stroke = false
	queue_redraw()


func lift_pen() -> void:
	_pending_new_stroke = true


func _draw() -> void:
	# Dark background so canvas is visible even when nothing is drawn
	var vp_size := get_viewport_rect().size
	draw_rect(Rect2(Vector2.ZERO, vp_size), bg_color)

	# Draw a subtle border
	draw_rect(Rect2(Vector2.ZERO, vp_size), Color(0.3, 0.3, 0.5), false, 4.0)

	var now := Time.get_ticks_msec() / 1000.0
	var prev = null

	for p in _points:
		var age: float = now - p.t
		var alpha: float = clampf(1.0 - age / trail_lifetime, 0.0, 1.0)
		var dot_col := stroke_color
		dot_col.a = alpha

		if prev != null and not p.new_stroke:
			var prev_age: float = now - prev.t
			var prev_alpha: float = clampf(1.0 - prev_age / trail_lifetime, 0.0, 1.0)
			var seg_col := stroke_color
			seg_col.a = (alpha + prev_alpha) * 0.5
			draw_line(prev.pos, p.pos, seg_col, stroke_width, true)

		draw_circle(p.pos, stroke_width * 0.5, dot_col)
		prev = p
