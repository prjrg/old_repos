module Course.Book.TautologyChecker where

import Course.Book.TypeDefinitions

data Prop = Const Bool | Var Char | Not Prop | And Prop Prop | Imply Prop Prop

type Subst = Assoc Char Bool

eval :: Subst -> Prop -> Bool
eval _ (Const b) = b
eval s (Var x) = find x s
eval s (Not p) = not (eval s p)
eval s (And p1 p2) = eval s p1 && eval s p2
eval s (Imply p1 p2) = eval s p1 <= eval s p2

vars :: Prop -> [Char]
vars (Const _) = []
vars (Var x) = [x]
vars (Not p) = vars p
vars (And p1 p2) = vars p1 ++ vars p2
vars (Imply p1 p2) = vars p1 ++ vars p2

type Bit = Int
int2bin :: Int -> [Bit]
int2bin 0 = []
int2bin n = n `mod` 2 : int2bin (n `div` 2)

bools :: Int -> [[Bool]]
bools n = map (reverse . map conv . make n . int2bin) range
              where
                range = [0..(2^n)-1]
                make n bs = take n (bs ++ repeat 0)
                conv 0 = False
                conv 1 = True

bools1 :: Int -> [[Bool]]
bools1 0 = []
bools1 n = map (False:) bss ++ map (True:) bss
              where bss = bools1 (n-1)


-- Remove duplicates from a list
rmdups :: Eq a => [a] -> [a]
rmdups [] = []
rmdups (x:xs) = x : filter (/=x) xs

substs :: Prop -> [Subst]
substs p = map (zip vs) (bools (length vs))
            where vs = rmdups (vars p)


-- Check if a proposition is a tautology
isTaut :: Prop -> Bool
isTaut p = and [eval s p | s <- substs p]