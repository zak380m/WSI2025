package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/debug"
	"strconv"
	"time"
)

var (
	BoardSize    int
	NumTiles     int
	GoalBoard    uint64
	MemoryLimit  int64 = 8 * 1024 * 1024 * 1024 
)

type Position uint8

const (
	Up = iota
	Right
	Down
	Left
)

var directionNames = []string{"UP", "RIGHT", "DOWN", "LEFT"}

type Heuristic struct {
	manhattanTable [][]int
	goalPositions  []Position
	directionDeltas []int
	heuristicFunc   func(uint64) int
}

func NewHeuristic(useLinearConflict bool) *Heuristic {
	h := &Heuristic{}
	h.directionDeltas = []int{-BoardSize, 1, BoardSize, -1}
	h.initManhattanTable()
	h.goalPositions = make([]Position, NumTiles)
	for tile := 1; tile < NumTiles; tile++ {
		h.goalPositions[tile] = Position(tile - 1)
	}

	if useLinearConflict {
        h.heuristicFunc = h.LinearConflict
    } else {
        h.heuristicFunc = h.ManhattanDistance
    }

	return h
}

func (h *Heuristic) initManhattanTable() {
	h.manhattanTable = make([][]int, NumTiles)
	for tile := 0; tile < NumTiles; tile++ {
		h.manhattanTable[tile] = make([]int, NumTiles)
		for pos := 0; pos < NumTiles; pos++ {
			if tile == 0 {
				h.manhattanTable[tile][pos] = 0
				continue
			}
			goalPos := tile - 1
			h.manhattanTable[tile][pos] = abs(pos%BoardSize-goalPos%BoardSize) + 
				abs(pos/BoardSize-goalPos/BoardSize)
		}
	}
}

func (h *Heuristic) ManhattanDistance(board uint64) int {
	distance := 0
	for pos := 0; pos < NumTiles; pos++ {
		shift := (NumTiles - 1 - pos) * 4
		tile := int((board >> shift) & 0xF)
		if tile != 0 {
			distance += h.manhattanTable[tile][pos]
		}
	}
	return distance
}

func (h *Heuristic) LinearConflict(board uint64) int {
	distance := h.ManhattanDistance(board)
	var grid []int

	grid = make([]int, NumTiles)
	for pos := 0; pos < NumTiles; pos++ {
		shift := (NumTiles - 1 - pos) * 4
		grid[pos] = int((board >> shift) & 0xF)
	}

	for y := 0; y < BoardSize; y++ {
		rowStart := y * BoardSize
		for x1 := 0; x1 < BoardSize-1; x1++ {
			pos1 := rowStart + x1
			tile := grid[pos1]
			if tile == 0 || (tile-1)/BoardSize != y {
				continue
			}
			for x2 := x1 + 1; x2 < BoardSize; x2++ {
				pos2 := rowStart + x2
				tile2 := grid[pos2]
				if tile2 != 0 && (tile2-1)/BoardSize == y && tile > tile2 {
					distance += 2
				}
			}
		}
	}

	for x := 0; x < BoardSize; x++ {
		for y1 := 0; y1 < BoardSize-1; y1++ {
			pos1 := y1*BoardSize + x
			tile := grid[pos1]
			if tile == 0 || (tile-1)%BoardSize != x {
				continue
			}
			for y2 := y1 + 1; y2 < BoardSize; y2++ {
				pos2 := y2*BoardSize + x
				tile2 := grid[pos2]
				if tile2 != 0 && (tile2-1)%BoardSize == x && tile > tile2 {
					distance += 2
				}
			}
		}
	}

	return distance
}

type PuzzleState struct {
	board   uint64
	empty   Position
	gScore  int16
	hScore  int16
	move    byte
	parent  *PuzzleState
}

func (s *PuzzleState) fScore() int {
	return int(s.gScore) + int(s.hScore)
}

func (s *PuzzleState) isGoal() bool {
	return s.board == GoalBoard
}

func (s *PuzzleState) getPath() []string {
	if s.parent == nil {
		return []string{}
	}
	return append(s.parent.getPath(), directionNames[s.move])
}

func (s *PuzzleState) moveEmpty(dir byte, h *Heuristic) *PuzzleState {
    delta := h.directionDeltas[dir]
    newPos := int(s.empty) + delta

    if newPos < 0 || newPos >= NumTiles {
        return nil
    }
    if (delta == -1 && int(s.empty)%BoardSize == 0) || 
       (delta == 1 && int(s.empty)%BoardSize == BoardSize-1) {
        return nil
    }

    emptyShift := (NumTiles - 1 - int(s.empty)) * 4
    newShift := (NumTiles - 1 - newPos) * 4
    tileValue := (s.board >> newShift) & 0xF

    newBoard := s.board
    newBoard &^= 0xF << emptyShift
    newBoard &^= 0xF << newShift
    newBoard |= tileValue << emptyShift

    hScore := int16(h.heuristicFunc(newBoard))

    return &PuzzleState{
        board:  newBoard,
        empty:  Position(newPos),
        gScore: s.gScore + 1,
        hScore: hScore,
        move:   dir,
        parent: s,
    }
}

type PriorityQueue struct {
	items []*PuzzleState
}

func (pq *PriorityQueue) Len() int {
	return len(pq.items)
}

func (pq *PriorityQueue) Push(x *PuzzleState) {
	pq.items = append(pq.items, x)
	i := len(pq.items) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if pq.items[i].fScore() >= pq.items[parent].fScore() {
			break
		}
		pq.items[i], pq.items[parent] = pq.items[parent], pq.items[i]
		i = parent
	}
}

func (pq *PriorityQueue) Pop() *PuzzleState {
	n := len(pq.items)
	item := pq.items[0]
	pq.items[0] = pq.items[n-1]
	pq.items = pq.items[:n-1]
	i := 0
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i
		if left < len(pq.items) && pq.items[left].fScore() < pq.items[smallest].fScore() {
			smallest = left
		}
		if right < len(pq.items) && pq.items[right].fScore() < pq.items[smallest].fScore() {
			smallest = right
		}
		if smallest == i {
			break
		}
		pq.items[i], pq.items[smallest] = pq.items[smallest], pq.items[i]
		i = smallest
	}
	return item
}

func isSolvable(board uint64) bool {
	inversions := 0
	emptyRow := 0

	for i := 0; i < NumTiles-1; i++ {
		shift := (NumTiles - 1 - i) * 4
		tilei := int((board >> shift) & 0xF)
		if tilei == 0 {
			emptyRow = i / BoardSize
			continue
		}
		for j := i + 1; j < NumTiles; j++ {
			shift := (NumTiles - 1 - j) * 4
			tilej := int((board >> shift) & 0xF)
			if tilej != 0 && tilei > tilej {
				inversions++
			}
		}
	}

	if BoardSize%2 == 1 {
		return inversions%2 == 0
	} else {
		return (emptyRow%2 == 1) == (inversions%2 == 0)
	}
}

func createTestPuzzle() uint64 {
	var board uint64
	numbers := []int{15, 11, 13, 12, 14, 10, 9, 5, 2, 6, 8, 1, 3, 7, 4, 0} //Miejsce na testową planszę
	for i, num := range numbers {
		board |= uint64(num) << ((NumTiles - 1 - i) * 4)
	}
	return board
}

func generateRandomPuzzle(minSteps int) uint64 {
	rand.Seed(time.Now().UnixNano())
	
	if minSteps > 0 {
		heuristic := NewHeuristic(false)
		state := &PuzzleState{
			board: GoalBoard,
			empty: Position(NumTiles - 1),
		}
		
		for i := 0; i < minSteps; i++ {
			var validMoves []byte
			for dir := byte(0); dir < 4; dir++ {
				if state.moveEmpty(dir, heuristic) != nil {
					validMoves = append(validMoves, dir)
				}
			}
			if len(validMoves) == 0 {
				break
			}
			dir := validMoves[rand.Intn(len(validMoves))]
			state = state.moveEmpty(dir, heuristic)
		}
		
		extraMoves := rand.Intn(400) + 1
		for i := 0; i < extraMoves; i++ {
			var validMoves []byte
			for dir := byte(0); dir < 4; dir++ {
				if state.moveEmpty(dir, heuristic) != nil {
					validMoves = append(validMoves, dir)
				}
			}
			if len(validMoves) == 0 {
				break
			}
			dir := validMoves[rand.Intn(len(validMoves))]
			state = state.moveEmpty(dir, heuristic)
		}
		
		return state.board
	} else {
		for {
			numbers := rand.Perm(NumTiles)
			board := packBoard(numbers)
			if isSolvable(board) {
				return board
			}
		}
	}
}

func packBoard(numbers []int) uint64 {
	var board uint64
	for i, num := range numbers {
		board |= uint64(num) << ((NumTiles - 1 - i) * 4)
	}
	return board
}

func findEmpty(board uint64) Position {
	for pos := 0; pos < NumTiles; pos++ {
		shift := (NumTiles - 1 - pos) * 4
		if (board>>shift)&0xF == 0 {
			return Position(pos)
		}
	}
	return 0
}

func solvePuzzle(initialState *PuzzleState, heuristic *Heuristic) ([]string, int) {
	openSet := PriorityQueue{}
	openSet.Push(initialState)

	closedSet := make(map[uint64]bool)
	visitedStates := 0

	// Add progress tracking
	lastPrintTime := time.Now()
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	// Channel to receive ticker events
	printChan := make(chan struct{})
	go func() {
		for range ticker.C {
			printChan <- struct{}{}
		}
	}()

	for openSet.Len() > 0 {
		select {
		case <-printChan:
			if openSet.Len() > 0 {
				current := openSet.items[0] // Peek at the best state without removing it
				fmt.Printf("\nProgress update after %v:\n", time.Since(lastPrintTime))
				fmt.Printf("Current best state - f-score: %d (g: %d, h: %d)\n", 
					current.fScore(), current.gScore, current.hScore)
				fmt.Printf("Visited states: %d, Open states: %d\n", visitedStates, openSet.Len())
				lastPrintTime = time.Now()
			}
		default:
			// Continue with normal processing
		}

		current := openSet.Pop()
		visitedStates++

		if current.isGoal() {
			return current.getPath(), visitedStates
		}

		if closedSet[current.board] {
			continue
		}
		closedSet[current.board] = true

		for dir := byte(0); dir < 4; dir++ {
			neighbor := current.moveEmpty(dir, heuristic)
			if neighbor == nil || closedSet[neighbor.board] {
				continue
			}
			openSet.Push(neighbor)
		}
	}

	fmt.Println("\nNo solution found")
	return nil, visitedStates
}

func abs(x int) int {
	return int(math.Abs(float64(x)))
}

func printBoard(board uint64) {
	for y := 0; y < BoardSize; y++ {
		for x := 0; x < BoardSize; x++ {
			pos := y*BoardSize + x
			shift := (NumTiles - 1 - pos) * 4
			value := (board >> shift) & 0xF
			if value == 0 {
				fmt.Print("   ")
			} else {
				fmt.Printf("%2d ", value)
			}
		}
		fmt.Println()
	}
}

func initBoardSize(size int) {
	BoardSize = size
	NumTiles = BoardSize * BoardSize
	
	GoalBoard = 0
	for i := 1; i < NumTiles; i++ {
		GoalBoard |= uint64(i) << ((NumTiles - 1 - (i-1)) * 4)
	}
}

func main() {
	debug.SetMemoryLimit(MemoryLimit)
	
	mode := 4
	if len(os.Args) > 1 {
		if arg, err := strconv.Atoi(os.Args[1]); err == nil && arg >= 3 && arg <= 5 {
			mode = arg
		}
	}
	
	//var minSteps int
	switch mode {
	case 3:
		initBoardSize(3)
		//minSteps = 0 
	case 4:
		initBoardSize(4)
		//minSteps = 21 
	case 5:
		initBoardSize(4)
		//minSteps = 0 
	default:
		fmt.Println("Invalid mode. Usage: program [3|4|5]")
		os.Exit(1)
	}
	
	manhattanHeuristic := NewHeuristic(false) 
    linearConflictHeuristic := NewHeuristic(true) 
	
	initialBoard := createTestPuzzle()
	fmt.Println("Initial board:")
	printBoard(initialBoard)
	fmt.Println()

	emptyPos := findEmpty(initialBoard)
	initialState := &PuzzleState{
		board:  initialBoard,
		empty:  emptyPos,
		gScore: 0,
		hScore: int16(manhattanHeuristic.ManhattanDistance(initialBoard)),
	}

	// fmt.Println("Solving with Manhattan distance heuristic...")
    // start := time.Now()
    // initialState.hScore = int16(manhattanHeuristic.ManhattanDistance(initialBoard))
    // solution, visited := solvePuzzle(initialState, manhattanHeuristic)
    // elapsed := time.Since(start)

	// fmt.Printf("Solution found in %d moves\n", len(solution))
	// fmt.Printf("Visited %d states\n", visited)
	// fmt.Printf("Time taken: %s\n", elapsed)
	// fmt.Println("Solution steps:", solution)
	// fmt.Println()

	fmt.Println("Solving with Linear Conflict heuristic...")
    start := time.Now()
    initialState.hScore = int16(linearConflictHeuristic.LinearConflict(initialBoard))
    solution, visited := solvePuzzle(initialState, linearConflictHeuristic)
    elapsed := time.Since(start)

	fmt.Printf("Solution found in %d moves\n", len(solution))
	fmt.Printf("Visited %d states\n", visited)
	fmt.Printf("Time taken: %s\n", elapsed)
	fmt.Println("Solution steps:", solution)
}