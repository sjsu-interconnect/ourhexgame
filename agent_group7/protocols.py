from attr import define, field
from abc import ABC, abstractmethod
from ourhexenv import OurHexGame


@define
class Agent(ABC):
    env: OurHexGame = field()

    def from_file(self, file) -> "Agent":
        pass

    def to_file(self, file: str) -> None:
        pass

    @abstractmethod
    def select_action(observation, reward, termination, truncation, info) -> int:
        pass
