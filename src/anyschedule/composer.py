from typing import Any, Optional
from .base import BaseScheduler, register_scheduler, registered_schedulers
from .scheduler.constant import ConstantScheduler


@register_scheduler("composer")
class Composer(BaseScheduler):
    def _sub_init(
        self,
        schedules: dict[str, Any] = {},
    ):
        self.steps_schedule = []
        st = 0
        final_value = self.value
        for idx, schedule in enumerate(schedules):
            if isinstance(schedule, dict):
                if "start" not in schedule:
                    schedule["start"] = st
                if "value" not in schedule:
                    schedule["value"] = final_value
                schedule_cls = registered_schedulers[schedule.pop("mode")]
                schedules[idx] = schedule_cls(**schedule)
            elif not isinstance(schedule, BaseScheduler):
                raise ValueError(
                    f"Only subclass of BaseScheduler or config dict can be used, got {schedule}"
                )
            current_schedule: BaseScheduler = schedules[idx]
            if current_schedule.init_value is None:
                current_schedule.init_value = final_value
            start = current_schedule.start
            end = current_schedule.end
            if isinstance(end, float):
                assert self.end != float("inf"), "end must be provided when use ratio end steps"
                current_schedule.set_end(self.end)
                end = current_schedule.end
            st = end
            final_value = current_schedule(end - 1)
            if self.steps_schedule and start > self.steps_schedule[-1][1]:
                self.steps_schedule.append((self.steps_schedule[-1][1], start, -1))
            self.steps_schedule.append((start, end, idx))
        self.steps_schedule.append((self.steps_schedule[-1][1], float("inf"), -1))
        self.schedules = schedules + [ConstantScheduler(value=self.value)]
        self.current_schedule = self.schedules[0]
        self.current_start = self.steps_schedule[0][0]
        self.current_end = self.steps_schedule[0][1]

    def find_range(self, step: int) -> int:
        left, right = 0, len(self.steps_schedule) - 1

        while left <= right:
            mid = (left + right) // 2
            start, end, idx = self.steps_schedule[mid]

            if start <= step < end:
                return start, end, idx
            elif step >= end:
                left = mid + 1
            else:
                right = mid - 1
        return -1, -1, -1

    def _get_value(self, step):
        if step < self.current_start or step >= self.current_end:
            self.current_start, self.current_end, idx = self.find_range(step)
            self.current_schedule = self.schedules[idx]
        return self.current_schedule(step)
