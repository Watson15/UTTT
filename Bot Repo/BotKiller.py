''' core imports '''
import numpy as np
import matplotlib.pyplot as plt
import queue

''' development imports'''
from time import perf_counter
from tqdm import tqdm

''' visualization imports '''
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgrey', 'white'])
import matplotlib.colors as mcolors
tab10_names = list(mcolors.TABLEAU_COLORS) # create a list of colours


# author: Aidan
# date: Nov 28 2023
# notes: Using line_completer.py as the baseline then added my own stuff in
# will look into using a tree search
# https://michaelxing.com/UltimateTTT/v3/ai/ bot seems to try and get two in a row for squares it is tring to get
# for squares it is not trying to get it puts them in bad squares to stop me from getting the square while also trying
# to land me in a bad sqaure to play in. 
# the AI seems to think the middle square in the middle is the best first move
# The best response to that is to put the other player into the corner square
# appears to be a good strategy to not repeat squares until you have to as it can allow the oponent to get two in a row
# dont make a move that will result in oponent putting you right back where you were when you dont want to be there
# all about wanting oponent to make forcing moves. Or a good move for them results in a better move for you
# maybe make a bot that tries to make a move that forces the opponent to make a move that is bad for them rather
# than making a move that is good for you
# the 9 major squares some or more valuable then others. Winning those squares is a priority
# The squares are ranked at start and can change based on game state

class BotKiller:
    '''
    1) tries to complete lines
    2) tries to block opponents
    3) it uses a probability distribution to preferentially play center and corner squares.
       the prob-dist is also used whenever there are multiple ways to complete/block a line
    '''
    MPQueue = queue.PriorityQueue() #put values in negative to get max
    #key is value given and the actual value is the move
    ''' ------------------ required function ---------------- '''
    
    def __init__(self,name: str = 'Chekhov') -> None:
        self.name = name
        # define the probability distribution
        self.box_probs = np.ones((3,3)) # edges
        self.box_probs[1,1] = 4 # center
        self.box_probs[0,0] = self.box_probs[0,2] = self.box_probs[2,0] = self.box_probs[2,2] = 2 # corners
        self.MajorSquareRank = np.array([[2,1,2],[1,3,1],[2,1,2]])#middle most valuable, then corners then edges
        #because 4 moves are made before start may be worth anaylizing the board and changing the rank of the squares
        self.GameState = np.zeros((3,3))#State of major boxes. (i.e. the total game state of the board)
    
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        return tuple(self.heuristic_mini_to_major(board_state = board_dict['board_state'],
                                                  active_box = board_dict['active_box'],
                                                  valid_moves = board_dict['valid_moves']))
    ''' --------- generally useful bot functions ------------ '''
    
    def _check_line(self, box: np.array) -> bool:
        '''
        box is a (3,3) array
        returns True if a line is found, else returns False '''
        for i in range(3):
            if abs(sum(box[:,i])) == 3: return True # horizontal
            if abs(sum(box[i,:])) == 3: return True # vertical

        # diagonals
        if abs(box.trace()) == 3: return True
        if abs(np.rot90(box).trace()) == 3: return True
        return False

    def _check_line_playerwise(self, box: np.array, player: int = None):
        ''' returns true if the given player has a line in the box, else false
        if no player is given, it checks for whether any player has a line in the box'''
        if player == None:
            return self._check_line(box)
        if player == -1:
            box = box * -1
        box = np.clip(box,0,1)
        return self._check_line(box)
    
    def pull_mini_board(self, board_state: np.array, mini_board_index: tuple) -> np.array:
        ''' extracts a mini board from the 9x9 given the its index'''
        temp = board_state[mini_board_index[0]*3:(mini_board_index[0]+1)*3,
                           mini_board_index[1]*3:(mini_board_index[1]+1)*3]
        return temp

    def get_valid(self, mini_board: np.array) -> np.array:
        ''' gets valid moves in the miniboard'''
        return np.where(mini_board == 0)

    def get_finished(self, board_state: np.array) -> np.array:
        ''' calculates the completed boxes'''
        opp_boxes = np.zeros((3,3))
        self_boxes = np.zeros((3,3))
        stale_boxes = np.zeros((3,3))
        # look at each miniboard separately
        for _r in range(3):
            for _c in range(3):
                mini_board = self.pull_mini_board(board_state, (_r,_c))
                self_boxes[_r,_c] = self._check_line_playerwise(mini_board, player = 1)
                opp_boxes[_r,_c] = self._check_line_playerwise(mini_board, player = -1)
                if sum(abs(mini_board.flatten())) == 9:
                    stale_boxes[_r,_c] = 1                   

        # return finished boxes (separated by their content)
        return (opp_boxes*-1, self_boxes, stale_boxes)
    
    def get_num_finished(self, board_state: np.array):
        ''' returns a list of the number of finished boxes for each player'''
        Boxes = [0,0,0]
        for r in range(3):
            for c in range(3):
                if(board_state[r,c] == 1):
                    Boxes[0]+=1
                elif(board_state[r,c] == -1):
                    Boxes[1]+=1
                else:
                    Boxes[2]+=1
        return Boxes

    def complete_line(self, mini_board: np.array) -> list:
        ''' completes a line if available '''
        # loop through valid moves with hypothetic self position there.
        # if it makes a line it's an imminent win
        imminent = list()
        valid_moves = self.get_valid(mini_board)
        
        for _valid in zip(*valid_moves):
            # create temp valid pattern
            valid_filter = np.zeros((3,3))
            valid_filter[_valid[0],_valid[1]] = 1
            if self._check_line(mini_board + valid_filter):
                imminent.append(_valid)
        return imminent
    
    def get_probs(self, valid_moves: list) -> np.array:
        ''' match the probability with the valid moves to weight the random choice '''
        probs = list()
        for _valid in valid_moves:
            probs.append(self.box_probs[_valid[0],_valid[1]])
        probs /= sum(probs) # normalize
        return probs
    
    def ChooseBestMajorSquare (self, board_state: np.array, active_box: tuple, valid_moves: list):
        ''' chooses the best major square to play in based on the current board state'''
        # if the active box is not the whole board
        if active_box != (-1,-1):
            # look just at the mini board
            mini_board = self.pull_mini_board(board_state, active_box)
            # look using the logic, select a move
            move = self.ChooseBestMiniSquare(mini_board)
            # project back to original board space
            return (move[0] + 3 * active_box[0],
                    move[1] + 3 * active_box[1])

        else:
            # use heuristic on finished boxes to select which box to play in
            imposed_active_box = self.major_heuristic(board_state)

            # call this function with the self-imposed active box
            return self.ChooseBestMajorSquare(board_state = board_state,
                                                active_box = imposed_active_box,
                                                valid_moves = valid_moves)

    def CalculateBoardState(self, board_state: np.array):
        ''' calculates the current state of the board for Major Square Heuristic'''
        for i in range(3):
            for j in range(3):
                self.GameState[i,j] = self.CalculateMiniBoardState(self.pull_mini_board(board_state, (i,j)))
                '''if (self.GameState[i,j] == 1):
                    self.MajorSquareRank[i,j] = 1
                elif(self.GameState[i,j] == -1):
                    self.MajorSquareRank[i,j] = -1
                elif(self.GameState[i,j] == 0):
                    self.MajorSquareRank[i,j] = 0
                else:
                    self.MajorSquareRank[i,j]+=self.GameState[i,j]#need to test this. See if its any good'''
                    #or do the following instead. GameState is for board state MSR is for value of major squares
                if (self.GameState[i,j] == 1 or self.GameState[i,j] == -1 or self.GameState[i,j] == 0):
                    #If a Major square is won, lost or stalemate then it is worth 0
                    self.MajorSquareRank[i,j] = 0
    def CalculateMiniBoardState(self, mini_board: np.array):
        ''' calculates the current state of the mini board for Major Square Heuristic'''
        #if the board is won return 1
        if self._check_line_playerwise(mini_board, player = 1):
            return 1
        #if the board is lost return -1
        if self._check_line_playerwise(mini_board, player = -1):
            return -1
        #if the board has more opponent moves in it return -0.5
        get_finished = self.get_num_finished(mini_board)
        #[0] is self, [1] is opponent, [2] is stale
        if get_finished[1]>get_finished[0]:
            return -0.5
        #if the board has more self moves in it return 0.5
        if(get_finished[1]<get_finished[0]):
            return 0.5
        #if the board is a stalemate return 0
        if(get_finished[2] == 9):
            return 0
        #if the board has same amount of opponent moves as self moves return 0.1
        return 0.1

    def ChooseBestMiniSquare (self, miniBoard: np.array):
        #look through the mini board and find the best square to play in based on what it does for me.
        #first check if can make a move that wins me a Major square. Then if it does check to see if winning
        #that square is valuable or not using CalculateMiniBoardState. If it is valuable play there. If not put in MP Queue
        #then check other possible moves and see what major square it puts them into. Check the value of those squares
        #using CalculateMiniBoardState. If it is valuable dont play there. Want to play a move that puts the opponent
        #in a bad major square. For each, put into the MPQueue. Once all moves are checked, take the move with the
        #highest value from the MPQueue and play it. 
        return (0,0)#Temp



    ''' ------------------ bot specific logic ---------------- '''
    
    def heuristic_mini_to_major(self, board_state: np.array,
                                active_box: tuple,
                                valid_moves: list) -> tuple:
        '''
        either applies the heuristic to the mini-board or selects a mini-board (then applies the heuristic to it)
        '''
        if active_box != (-1,-1):
            # look just at the mini board
            mini_board = self.pull_mini_board(board_state, active_box)
            # look using the logic, select a move
            move = self.mid_heuristic(mini_board)
            # project back to original board space
            return (move[0] + 3 * active_box[0],
                    move[1] + 3 * active_box[1])

        else:
            # use heuristic on finished boxes to select which box to play in
            imposed_active_box = self.major_heuristic(board_state)

            # call this function with the self-imposed active box
            return self.heuristic_mini_to_major(board_state = board_state,
                                                active_box = imposed_active_box,
                                                valid_moves = valid_moves)

    def major_heuristic(self, board_state: np.array) -> tuple:
        '''
        determines which miniboard to play on
        note: having stale boxes was causing issues where the logic wanted to block
              the opponent but that mini-board was already finished (it was stale)
        '''
        z = self.get_finished(board_state)
        # finished boxes is a tuple of 3 masks: self, opponent, stale 
        self_boxes  = z[0]
        opp_boxes   = z[1]
        stale_boxes = z[2]
        
        # identify imminent wins for self
        imminent_wins = self.complete_line(self_boxes)
        
        # make new list to remove imminent wins that point to stale boxes
        stale_boxes = zip(*np.where(stale_boxes))
        for stale_box in stale_boxes:
            if stale_box in imminent_wins:
                imminent_wins.remove(stale_box)
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins), p=self.get_probs(imminent_wins))]
        
        # attempt to block
        imminent_wins = self.complete_line(opp_boxes)
        # make new list to remove imminent wins that point to stale boxes
        stale_boxes = zip(*np.where(stale_boxes))
        for stale_box in stale_boxes:
            if stale_box in imminent_wins:
                imminent_wins.remove(stale_box)
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins), p=self.get_probs(imminent_wins))]

        # else take random
        internal_valid = list(zip(*self.get_valid(self_boxes + opp_boxes)))
        return internal_valid[np.random.choice(len(internal_valid), p=self.get_probs(valid_moves))]
        
    def mid_heuristic(self, mini_board: np.array) -> tuple:
        ''' main mini-board logic
        attempt to finish the box (biases to center and corners)
        attempt to block opponent (again, with bias)
        randomly take a move (with bias)
        '''
        # try to complete a line on this miniboard
        imminent_wins = self.complete_line(mini_board)
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins))]

        ''' attempt to block'''
        imminent_wins = self.complete_line(mini_board * -1) # pretend to make lines from opponent's perspective
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins))]

        # else play randomly
        valid_moves = list(zip(*self.get_valid(mini_board)))
        return valid_moves[np.random.choice(len(valid_moves), p=self.get_probs(valid_moves))]

