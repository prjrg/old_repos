module Main where

--import Course.Book.CountdownProblem
import Course.Book.TicTacToe
import System.IO

--main :: IO ()
--main = print (solutions1 [1,3,7,10,25,50] 765)

--main = print (solutions1 [1,3,7,10,25,50] 831)

-- main :: IO ()
-- main = tictactoe

-- TicTacToe Human vs Computer (Finally)
main :: IO ()
main = do hSetBuffering stdout NoBuffering
          play empty O
