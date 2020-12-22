module Course.Book.InteractiveProgrammingLife where

{-
A game that models a simple evolutionary system based on cells, and is played on a 2-dimensional board
-}

{-
Rules:
  The next generation of the board is given by
  1. A living cell survives if it has precisely 2 or 3 neighbouring squares containing living cells,
  2. An empty square gives birth to a living cell if it has precisely 3 neighbours that contain living cells, and remains
 empty otherwise.
-}

-- Screen utilities:

-- Clears screen
cls :: IO ()
cls = putStr "\ESC[2J"


type Pos = (Int, Int)

writeat :: Pos -> String -> IO ()
writeat p xs = do goto p
                  putStr xs

goto :: Pos -> IO ()
goto (x, y) = putStr ("\ESC[" ++ show y ++ ";" ++ show x ++ "H")

-- Game of life implementation

width :: Int
width = 10

height :: Int
height = 10

type Board = [Pos]

glider :: Board
glider = [(4,2), (2,3), (4,3), (3,4), (4,4)]

showcells :: Board -> IO ()
showcells b = sequence_ [writeat p "O" | p <- b]

isAlive :: Board -> Pos -> Bool
isAlive b p = p `elem` b

isEmpty :: Board -> Pos -> Bool
isEmpty b p = not (isAlive b p)

neighbs :: Pos -> [Pos]
neighbs (x,y) = map wrap [(x-1, y-1), (x, y-1),
                          (x+1, y-1), (x-1, y),
                          (x+1, y), (x-1, y+1),
                          (x, y+1), (x+1, y+1)]

wrap :: Pos -> Pos
wrap (x, y) = (((x-1) `mod` width) + 1, ((y-1) `mod` height) + 1)

liveneighbs :: Board -> Pos -> Int
liveneighbs b = length . filter (isAlive b) . neighbs

survivors :: Board -> [Pos]
survivors b = [p | p <- b, liveneighbs b p `elem` [2,3]]

births :: Board -> [Pos]
births b = [(x,y) | x <- [1..width], y <- [1..height], isEmpty b (x,y), liveneighbs b (x,y) == 3]

births1 :: Board -> [Pos]
births1 b = [p | p <- rmdups (concat (map neighbs b)), isEmpty b p, liveneighbs b p == 3]

rmdups :: Eq a => [a] -> [a]
rmdups [] = []
rmdups (x:xs) = x : rmdups (filter (/= x) xs)

nextgen :: Board -> Board
nextgen b = survivors b ++ births b

life :: Board -> IO ()
life b = do cls
            showcells b
            wait 500000
            life (nextgen b)

wait :: Int -> IO ()
wait n = sequence_ [return () | _ <- [1..n]]
