import numpy as np

class BitSmartAgent:
    def select_action(self, env, info):
        '''
            1. Prioritize moves closer to the center.
            2. Prioritize moves connecting agent to cells of its own color
            3. Prioritize moves which enables multiple connection opportunities
            4. Based on the agent direction
        '''

        # available_moves = env.action_space(agent_id).sample(info["action_mask"])
        available_moves = [i for i, valid in enumerate(info["action_mask"]) if valid]

        board = env.board
        board_size = board.shape[0]
        
        if len(available_moves) == 0:
            return None
            
        scores = []
        center = board_size // 2
        player_color = board[0, 0] if np.any(board != 0) else 1
        
        for move in available_moves:
            row = move // board_size
            col = move % board_size
            score = 0
            
            #  Prioritize moves closer to the center
            distance_to_center = abs(row - center) + abs(col - center)
            score -= distance_to_center * 1.5
            
            potential_connections = 0
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue # no change
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < board_size and 0 <= new_col < board_size):
                        if board[new_row, new_col] == player_color:
                            score += 5 # Prioritize moves connecting agent to cells of its own color
                        elif board[new_row, new_col] == 0:
                            potential_connections += 1
            
            # Prioritize moves which enables multiple connection opportunities
            score += potential_connections * 0.8
            
            # Based on the agent direction
            if player_color == 1:  # Needs top-bottom connection
                progress_score = (board_size - 1 - abs(row - (board_size - 1))) * 0.75
                score += progress_score
            else:  # Needs left-right connection
                progress_score = (board_size - 1 - abs(col - (board_size - 1))) * 0.75
                score += progress_score
            scores.append(score)
        
        # Choose move with highest score
        best_move_idx = np.argmax(scores)
        return available_moves[best_move_idx]