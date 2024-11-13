# Our Hex Game
We are going to create a common Hex Game environment ```OurHexGame``` for our PA5 and final project.

## Assumption
- The board size is 11x11.

## Agents
- possible_agents ```[“player_1”, “player_2”]```
- “player_1”: red, vertical
- “player_2”: blue, horizontal

## Observation 
- Dict {
  - Box(board_size, board_size)
  - Discrete(1) # pie rule used if 1 else 0}
- Values - 0: empty, 1: player_1, 2: player_2
- Use the cell ID based on the first image on https://en.wikipedia.org/wiki/Hex_(board_game) (e.g., 1A, 1B, ...)

## Info
- horizontal (0) or vertical (1)
- The environment should provide an action mask to indicate invalid actions. (We can repurpose the observation.）
  - In the obs, if a hex is marked with 1 or 2, the mask should be 0. (invalid)
  - Otherwise, the mask is 1. (valid)
  - pie rule
  
## Action
- Discrete(board_size x board_size + 1)
  - Line 1: 1A (0), 1B, ...., 1K (10)
  - Line 2: 2A (11), 1B, ...., 2K (21)
  - ....
  - the last action is the pie rule 


## Reward Sparse
- define ```sparse_flag``` to turn on/off the sparse reward env. If False, it should use the dense reward env.
- Win +1
- Lose -1
- Otherwise, 0

## Reward Dense
- Each step, -1
- Win +floor((board_size * board_size)/2)
- Lose -ceil((board_size * board_size)/2)

## Termination
- (run DFS to check the winner after each cycle) If there is a winner, terminate
- If illegal move, terminate (reset)
  - Agent should not try the illegal actions!

## Rendering
please someone donate your code! thank you!

# Runner
- See ```myrunner-eg.py```

```python
agent = GXXAgent(env)
...
action = agent.select_action(observation, reward, termination, truncation, info)
```