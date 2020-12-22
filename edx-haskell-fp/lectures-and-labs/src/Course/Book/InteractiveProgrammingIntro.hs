module Course.Book.InteractiveProgrammingIntro where

-- Basic actions
-- getChar : reads a Char from the keyboard
-- putChar : puts a Char into the screen - dual of getChar
-- return v :: a -> IO a: it returns the value v without performing any interaction with the user

-- Sequencing
-- The do notation: composing actions
{-
do v1 <- a1
   v2 <- a2
   .
   .
   .
   vn <- an
   return (f v1 v2 ... vn)
-}

-- Action reading three characters by discarding the second
act :: IO (Char, Char)
act = do x <- getChar
         getChar
         y <- getChar
         return (x,y)


-- Derived primitives
getLine1 :: IO String
getLine1 = do x <- getChar
              if x == '\n' then return []
              else
                do xs <- getLine1
                   return (x:xs)

putStr1 :: String -> IO ()
putStr1 [] = return ()
putStr1 (x:xs) = do putChar x
                    putStr1 xs

putStrLn1 :: String -> IO ()
putStrLn1 xs = do putStr1 xs
                  putChar '\n'


strlen :: IO ()
strlen = do putStr "Enter a string: "
            xs <- getLine1
            putStr "The string has "
            putStr (show (length(xs)))
            putStr " characters"

