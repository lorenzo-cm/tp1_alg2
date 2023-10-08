from utils import Point, Segment, orientation
import heapq

class Event:
    def __init__(self, point: Point, segment: Segment, is_left: bool) -> None:
        self.point = point
        self.segment = segment
        self.is_left = is_left

    def __lt__(self, other: 'Event') -> bool:
        return self.point.x < other.point.x or (self.point.x == other.point.x and self.is_left)

def do_segments_intersect(s1: Segment, s2: Segment) -> bool:
    # Check if bounding boxes intersect
    if (max(s1.points[0].x, s1.points[1].x) < min(s2.points[0].x, s2.points[1].x) or
        max(s2.points[0].x, s2.points[1].x) < min(s1.points[0].x, s1.points[1].x) or
        max(s1.points[0].y, s1.points[1].y) < min(s2.points[0].y, s2.points[1].y) or
        max(s2.points[0].y, s2.points[1].y) < min(s1.points[0].y, s1.points[1].y)):
        return False

    # Check orientations
    o1 = orientation(s1.points[0], s1.points[1], s2.points[0])
    o2 = orientation(s1.points[0], s1.points[1], s2.points[1])
    o3 = orientation(s2.points[0], s2.points[1], s1.points[0])
    o4 = orientation(s2.points[0], s2.points[1], s1.points[1])

    return o1 != o2 and o3 != o4

def sweep_line_intersection(segments: list[Segment]) -> bool:
    events: list[Event] = []
    for segment in segments:
        left, right = sorted(segment.points, key=lambda p: p.x)
        events.append(Event(left, segment, True))
        events.append(Event(right, segment, False))

    heapq.heapify(events)

    active_segments: set[Segment] = set()

    while events:
        event = heapq.heappop(events)
        if event.is_left:
            for active_segment in active_segments:
                if do_segments_intersect(event.segment, active_segment):
                    return True
            active_segments.add(event.segment)
        else:
            active_segments.remove(event.segment)

    return False