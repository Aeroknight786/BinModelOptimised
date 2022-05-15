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

We define and empty array to fill with the spot price values on the terminal exercise date. We back calculate from the lowest possible value by multiplying the UpMove and dividing by DownMove at each step. This helps us iterate through all the possible terminal values considering all paths stemming from the binomial distribution. We then calculate the boundary conditions on the terminal date by calculating max(0, S[j] - K) where K is the strike price and S is the precalculated terminal spot price.

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


```python
  #Step backwards through Tree
  for i in np.arange(N,0,-1):
    for j in range(0,i):
      C[j] = discount * ( q*C[j+1] + (1-q)*C[j] )
  
  return C[0]

binomial_tree(K,T,So,r,N,u,d,opttype = 'C')
```
```
```
