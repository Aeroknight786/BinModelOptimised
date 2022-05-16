# Binomial Option Pricing in Python
We here implement a binomial option pricing model derived under risk neutral no-arbitrage conditions based on Chapter 10 (Options, Futures and Other Derivatives, John.C.Hull). First we price a vanilla call option using two implementation methods, one using basic for-loops to traverse the tree and a optimal vectorised technique for more efficiency. Firstly, we define the input parameters.
```python
#Initialising Parameters:

So = 100          #Initial Stock Price
K = 100           #Strike Price
T = 1             #Time Left to Maturity in Years
r = 0.06          #Annual Risk Free Rate
N = 3             #No. of Time Steps
u = 1.1           #Spot Price Up-Factor in Binomial Model
d = 1/u           #Ensure a Recombining Symmetrical Tree
opttype = 'C'     #Option Type 'C' or 'P' for Call or Put Option
H = 150           #Up and Out Barrier for Barrier Option
```

Here we define constants required for the tree construction such as the time increment at each step, the risk-neutral probability of price movement and the discount factor to be multiplied from the future payoff. 

```python
def binomial_tree(K,T,So,r,N,u,d,opttype = 'C'):
  #Pre-Computing Constants
  dt = T/N                          #Total Time to Maturity divided by number of periods.
  q = (np.exp(r * dt) - d)/(u - d)  #Risk Neutral Formula
  discount = np.exp(-r * dt)        #Discounting the Expected Pay-Offs when exercised.
```

We define and empty array to fill with the spot price values on the terminal exercise date. We back calculate from the lowest possible value by multiplying the UpMove and dividing by DownMove at each step. This helps us iterate through all the ponsible terminal values considering all paths stemming from the binomial distribution. We then calculate the boundary conditions on the terminal date by calculating max(0, S[j] - K) where K is the strike price and S is the precalculated terminal spot price.

```python
  #Initialise Asset Prices, Time Step N
  S = np.zeros(N + 1)
  S[0] = So * (d**N)
  for j in range (1 , N+1):
    S[j] = S[j-1] * u/d
  
  #Initialise Option Values at Maturity
  C = np.zeros(N + 1)               #4 different possible pay offs if N = 3
  for j in range(0,N+1):
    C[j] = max(0, S[j] - K)

```
In this section, we iterate across the nodes of the tree in order to calculate the risk neutral option premium at time T = 0. 2 for-loops are used for this process, the first one being an iterator from the last time step layer(N) to the 0th time step. The second for-loop iterates from the lowest valued node of layer i, to the highest valued node according to the binomial path. Every layer i contains (i+1) nodes. Hence we calculate the value of C[j] using C[j] and C[j+1]. For example, in a 2 period binomial model, C[1,0] = func(C[2,0], C[2,1]). This means that when we update the value of C[0] it remains unchanged for the rest of the iteration along the nodes of layer[1]. When calculating C[0,0], we use C[1,0] and C[1,1]. The terminal value of the option is C[0] at the end of all iterations. 

```python
  #Step backwards through Tree
  for i in np.arange(N,0,-1):
    for j in range(0,i):
      C[j] = discount * ( q*C[j+1] + (1-q)*C[j] )
  
  return C[0]

binomial_tree(K,T,So,r,N,u,d,opttype = 'C')
```
We can see that this particular algorithm has On^2 complexity and for recommended accuracy levels, we'd have trouble with the amount of computing time, consumed for each set of parameters. Hence, we need to optimise the process by using numpy vectors as at the vectorised level, we can compute much faster for higher values of N.
Here we used numpy vectors to quicky calculate the values of the terminal spot prices and option payoffs at expiry instead of two separate for-loops.
```python
  #Initialise Option Values at Maturity
  S = So * d** (np.arange(N,-1,-1)) * u **(np.arange(0,N+1,1))

  #Initialise Option Values
  C = np.maximum(S - K, np.zeros(N+1))
```
Finally for iterating across and computing the value of the nodes, we use only one for-loop to iterate through the layers, while we use two shifted subsets of the Node Vectors to quickly compute the values of nodes on the preceding layer.
```python
#Step backwards through Tree
  for i in np.arange(N,0,-1):
    C = discount * (q * C[1:i+1] + (1 - q) * C[0:i])
  return C[0]
binomial_tree_vectorised(K,T,So,r,N,u,d,opttype = 'C')
```
We've prepared this code in order to perform properly for a Vanilla Call option, and on plotting for increasing values of time steps, we see that it converges. Here the value of U and D remains constant even though we are increasing the number of time steps. This however is incorrect as according to a Cox, Ross and Rubenstein initialisation, we will have U = e^(sigma * sqrt(T)). Initialisation will be covered shortly.


![image](https://user-images.githubusercontent.com/51220035/168601635-a0f21b78-1522-41c5-8631-f47e5a8d9cbb.png)




```
```
