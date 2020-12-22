module Course.Book.TicTacToe where

import Data.Char
import Data.List
import System.IO

--Tic Tac Toe -- noughts and crosses

size :: Int
size = 3

type Grid = [[Player]]

data Player = O | B | X deriving (Eq, Ord, Show)

next :: Player -> Player
next O = X
next B = B
next X = O

-- Grid utilities
empty :: Grid
empty = replicate size (replicate size B)

full :: Grid -> Bool
full = all (/= B) . concat

turn :: Grid -> Player
turn g = if os <= xs then O else X
          where
            os = length (filter (== O) ps)
            xs = length (filter (== X) ps)
            ps = concat g

wins :: Player -> Grid -> Bool
wins p g = any line (rows ++ cols ++ dias)
           where
            line = all (== p)
            rows = g
            cols = transpose g
            dias = [diag g, diag (map reverse g)]

-- Returns the main diagonal of a grid
diag :: Grid -> [Player]
diag g = [g !! n !! n | n <- [0..size-1]]

won :: Grid -> Bool
won g = wins O g || wins X g

-- Displaying the grid
putGrid :: Grid -> IO ()
putGrid = putStrLn . unlines . concat . interleave bar . map showRow
          where bar = [replicate ((size*4)-1) '-']

showRow :: [Player] -> [String]
showRow = beside . interleave bar . map showPlayer
          where
            beside = foldr1 (zipWith (++))
            bar = replicate 3 "|"

showPlayer :: Player -> [String]
showPlayer O = ["   ", " O ", "   "]
showPlayer B = ["   ", "   ", "   "]
showPlayer X = ["   ", " X ", "   "]

interleave :: a -> [a] -> [a]
interleave x [] = []
interleave x [y] = [y]
interleave x (y:ys) = y : x : interleave x ys

-- Making a move
valid :: Grid -> Int -> Bool
valid g i = 0 <= i && i < size^2 && concat g !! i == B

move :: Grid -> Int -> Player -> [Grid]
move g i p = if valid g i then [chop size (xs ++ [p] ++ ys)] else []
              where (xs, B:ys) = splitAt i (concat g)

chop :: Int -> [a] -> [[a]]
chop n [] = []
chop n xs = take n xs : chop n (drop n xs)


-- Reading a number
-- Reading a grid index number
getNat :: String -> IO Int
getNat prompt = do putStr prompt
                   xs <- getLine
                   if xs /= [] && all isDigit xs then return (read xs)
                   else do putStrLn "ERROR: Invalid number"
                           getNat prompt

-- Human vs Human
tictactoe :: IO ()
tictactoe = run empty O

type Pos = (Int, Int)
-- Clears screen
cls :: IO ()
cls = putStr "\ESC[2J"

goto :: (Int,Int) -> IO ()
goto (x,y) = putStr ("\ESC[" ++ show y ++ ";" ++ show x ++ "H")

run :: Grid -> Player -> IO ()
run g p = do cls
             goto (1,1)
             putGrid g
             run' g p

run' :: Grid->Player-> IO ()
run' g p | wins O g = putStrLn "Player O wins!\n"
         | wins X g = putStrLn "Player X wins!\n"
         | full g   = putStrLn "It's a draw!\n"
         | otherwise =
                do i <- getNat (prompt p)
                   case move g i p of
                    [] -> do putStrLn "Error: Invalid move"
                             run' g p
                    [g'] -> run g' (next p)

prompt :: Player -> String
prompt p = "Player " ++ show p ++ ", enter your move: "

--Game trees
data Tree a = Node a [Tree a] deriving Show

gametree :: Grid -> Player -> Tree Grid
gametree g p = Node g [gametree g' (next p) | g' <- moves g p]

moves :: Grid -> Player -> [Grid]
moves g p
  | won g = []
  | full g = []
  | otherwise = concat [move g i p | i <- [0..((size^2)-1)]]

-- Pruning the game tree to avoid excessive nodes (by depth)
prune :: Int -> Tree a -> Tree a
prune 0 (Node x _) = Node x []
prune n (Node x ts) = Node x [prune (n-1) t | t <- ts]

depth :: Int
depth = 9

-- Minimax algorithm
minimax :: Tree Grid -> Tree (Grid, Player)
minimax (Node g [])
          | wins O g  = Node (g, O) []
          | wins X g  = Node (g, X) []
          | otherwise = Node (g, B) []
minimax (Node g ts)
          | turn g == O = Node (g, minimum ps) ts'
          | turn g == X = Node (g, maximum ps) ts'
                          where
                            ts' = map minimax ts
                            ps  = [p | Node (_, p) _ <- ts']

-- Next best move for a given player
-- :: We first build the tree up to the specified max explored depth and then apply minimax to that subtree by choosing the
-- grid that the player label is the same as the root node
bestmove :: Grid -> Player -> Grid
bestmove g p = head [g' | Node (g', p') _ <- ts, p' == best]
              where
                tree = prune depth (gametree g p)
                Node (_, best) ts = minimax tree


-- Human vs Computer (Finally)
-- main :: IO ()
-- main = do hSetBuffering stdout NoBuffering
--           play empty O

play :: Grid -> Player -> IO ()
play g p = do cls
              goto (1,1)
              putGrid g
              play' g p

play' :: Grid -> Player -> IO ()
play' g p
    | wins O g = putStrLn "Player O wins!\n"
    | wins X g = putStrLn "Player X wins!\n"
    | full g   = putStrLn "It's a draw!\n"
    | p == O   = do i <- getNat (prompt p)
                    case move g i p of
                        [] -> do putStrLn "ERROR: Invalid move"
                                 play' g p
                        [g'] -> play g' (next p)
    | p == X   = do putStr "Player X is thinking... "
                    (play $! (bestmove g p)) (next p)

