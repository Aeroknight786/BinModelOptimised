# Binomial Option Pricing in Python
We here implement a binomial option pricing model derived under risk neutral no-arbitrage conditions based on Chapter 10 (Options, Futures and Other Derivatives, John.C.Hull) and the Binomal Option pricing playlist by ASX Portfolio YT. First we price a vanilla call option using two implementation methods, one using basic for-loops to traverse the tree and a optimal vectorised technique for more efficiency. Firstly, we define the input parameters.
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
We've prepared this code in order to perform properly for a Vanilla Call option, and on plotting for increasing values of time steps, we see that it converges but also increases by a few orders which is inaccurate. Here, in the implementation the value of U and D remains constant even though we are increasing the number of time steps. This however is incorrect as according to a Cox, Ross and Rubenstein initialisation, we will have U = e^(sigma * sqrt(T)). Initialisation will be covered shortly.


![image](https://user-images.githubusercontent.com/51220035/168601635-a0f21b78-1522-41c5-8631-f47e5a8d9cbb.png)

We have also not added functionality in our code for valuing Put Options. We add the following changes to our previous code in order to make the code more robust to the type of option fed as a parameter. The sole change is during the initialisation of the terminal payoffs, and we add a conditional statement to deal with the same.

```python
  C = np.zeros(N+1)
  for j in range(0,N+1):
    if (opttype == 'C'):
      C[j] = max(0, S[j] - K)
    else:
      C[j] = max(0, K - S[j])
```

Until now, we have prepared a function capable of valuing European Options exclusively; options that can be exercised on the terminal date only. However American Options are also widely traded and this model is capable of computing options with such payoffs. Whenever we compute the value of a node in the tree, we treat that node as if we have an option to exercise it at that moment in time. We have a price for the premium required to be paid and a possible payoff for the same on the basis of the paths possible from that particular node. The payoff for exercising it at that moment(+/-|S-K|) can be compared with the payoff if we held the option going ahead. We treat every node as if it is the first node and we compare the payoff of holding against that of exercising. We implement the same in the following section using 2 for-loops.

```python
  for i in np.arange(N-1,-1,-1):
    for j in range(0,i+1):
      S = So * (u**j) * (d**(i-j))
      #Here we calculate Spot Prices for each individual layer again so we can recompute payoff.
      C[j] = discount * (q * C[j+1] + (1 - q) * C[j])
      
      #After computing payoffs from future nodes, we compare against payoff for immediate exercise.
      if opttype == 'P':
        C[j] = max(C[j], K - S)
      else:
        C[j] = max(C[j], S - K)

  return C[0]
```
Now, we move on to a faster NumPy optimised implementation. Spot price and Payoffs need to computed at each step as we have to find if exercise is optimal at that time step. As we have to start iterating from the second-last layer and each layer and (i+1) number of nodes, we have the range for i beginning from N-1. The truncation of the payoff vector C is done so it can be readily be compared with the spot price vector that is recalculated at every layer iteration.

```python
  for i in np.arange(N-1,-1,-1):
    S = So* (u**(np.arange(0,i+1,1))) * (d**(np.arange(i,-1,-1)))
    C[:i+1] = discount * (q * C[1:i+2] + (1-q) * C[0:i+1])
    C = C[:-1]          # We truncate the last value as the spot price vector needs to be the same size as the payoff vector.
    
    
    if opttype == 'P':
      C = np.maximum(C, K - S)
    else:
      C = np.maximum(C, S - K)

  return C[0]
```
We also add implementations for valuing Barrier Options, in this case, an up-and-out Put Barrier Option. The changes made are essentially another filter on the Terminal Payoff to replace the payoffs beyond the spot price barrier by zero. We also need to calculate whether the barrier is exceeded by the spot price at every node, due to which there is a necessity to compute the spot price for every layer.

```python
  #Check Terminal Condition PayOff 
  for j in range(0,N+1):
    S = So * (u**j) * (d**(N-j))
    if S>=H:
      C[j] = 0

  #Backward Iteration Through the Tree
  for i in np.arange(N-1, -1,-1):
    for j in range(0,i+1):
      S = So* (u**j) * (d**(i-j))
      if S >= H:
        C[j] = 0
      else:
        C[j] = discount * (q*C[j+1] + (1-q)*C[j])
  
  return C[0]
```

Similar to the optimized implementation of the American Option Binomial Tree, the code now uses another conditional statement after the value of the node is computed to reduce the value of the payoff to 0 at those indices where [S >= H] as defined by the barrier function. 
```python
  #Check Terminal Condition PayOff
  C[S >= H ] = 0

  #Backward Recursion through the Tree
  for i in np.arange(N-1, -1, -1):
    S = So* (u**(np.arange(0,i+1,1))) * (d**(np.arange(i,-1,-1)))
    C[:i+1] = discount * (q * C[1:i+2] + (1-q) * C[0:i+1])
    C = C[:-1]
    C[S >= H] = 0
  
  return C[0]
```
Until now, the value of u and d used was arbitrary and did not take into account the size of the time step or the volatility of the underlying, both important factors in traditional models such as the BSM model. We explore two models here, the first one being the Cox, Ross and Rubinstein model that prepares a recombining tree whose values are derived by using the formula of expectation return value of the tree and equating against the variance in the time step of the underlying's GBM. The changes included in the beginning of the function are:
![image](https://user-images.githubusercontent.com/51220035/168925196-725013dc-9040-4824-ad94-6398a4b95796.png)
![image](https://user-images.githubusercontent.com/51220035/168925213-8f797414-ccbd-417b-b3c2-61a65f759fc8.png)


```python
def CRR_Method(K,T,So,r,N,sigma,opttype = 'C'):
  #Pre-Computing Constants
  dt = T/N                          #Total Time to Maturity divided by number of periods.
  u = np.exp(sigma*np.sqrt(dt))
  d = 1/u
  q = (np.exp(r * dt) - d)/(u - d)  #Risk Neutral Formula
  discount = np.exp(-r * dt)        #Discounting the Expected Pay-Offs when exercised.
```
We now finally calculate U and D based on the Jarrow and Rudd method that equates the Expectation Value of Return against the Mean drift term of the GBM Process and the Expected Variance against the Volatility term of the GBM. We get a risk neutral probability = 1/2 in this case. The implementation is as follows.
![image](https://user-images.githubusercontent.com/51220035/168925983-47a10b3f-f88b-4143-83b7-463ead610c20.png)

```python
def JR_Method(K,T,So,r,N,sigma,opttype = 'C'):
  #Pre-Computing Constants
  dt = T/N
  nu = r - 0.5*sigma**2
  u = np.exp(nu*dt + sigma*np.sqrt(dt))
  d = np.exp(nu*dt - sigma*np.sqrt(dt))
  q = 0.5
  discount = np.exp(-r * dt)        #Discounting the Expected Pay-Offs when exercised.
```
