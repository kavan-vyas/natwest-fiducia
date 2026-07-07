# fiducia - credit risk weighting methodology

this doc explains how the category weights used in the fiducia credit risk formula were derived, and lists the factors considered with a quick reason for each.

---

## 1. method: weighted scoring model (SMART)

the weights come from a **weighted scoring model**, a simplified form of SMART (simple multi attribute rating technique). each category gets two independant scores, multiplied together then normalised.

**step 1: importance score**

each category $i$ is rated 1 to 5 for how much it should matter to credit risk, based on reasoning about affordability, stability, and repayment capacity:

$$\text{importance}_i \in \{1, 2, 3, 4, 5\}$$

**step 2: reliability multiplier**

each category also gets a reliability score. fields the user reports as a hard figure (salary, mortgage, savings) count as verified. fields relying on the user honestly self reporting somthing sensitive (missed payments, credit history length, recent applications) get discounted, since they cant be checked against a real credit bureau:

$$
\text{reliability}_i =
\begin{cases}
1.0 & \text{verified input} \\
0.6 & \text{self declared input}
\end{cases}
$$

**step 3: raw score**

$$\text{raw}_i = \text{importance}_i \times \text{reliability}_i$$

**step 4: normalise to percentages**

$$w_i = \frac{\text{raw}_i}{\displaystyle\sum_{j=1}^{n} \text{raw}_j}$$

where $n = 9$, the total number of categorys. this guarantees $\sum_i w_i = 1$.

**why this method still holds up**

- every weight traces back to two stated numbers, not a random guess
- the reliability multiplier is the actual mechanism that explains why a self declared field carries less weight than its raw importance suggests
- its a real, recognised multi criteria scoring technique, not something made up for this coursework

---

## 2. worked calculation

with $n = 9$ categories:

$$\sum_{j=1}^{9} \text{raw}_j = 5.0 + 4.0 + 3.0 + 3.0 + 1.8 + 2.0 + 2.0 + 1.0 + 0.6 = 22.4$$

| category                              | importance (1-5) | reliability | raw score | weight  |
|----------------------------------------|:-----------------:|:-----------:|:---------:|:-------:|
| affordability (DTI)                    | 5                  | 1.0         | 5.0       | 22.32%  |
| employment stability                   | 4                  | 1.0         | 4.0       | 17.86%  |
| self declared payment history          | 5                  | 0.6         | 3.0       | 13.39%  |
| savings buffer & trend                 | 3                  | 1.0         | 3.0       | 13.39%  |
| self declared credit history length    | 3                  | 0.6         | 1.8       | 8.04%   |
| housing status                         | 2                  | 1.0         | 2.0       | 8.93%   |
| debt composition (credit mix)          | 2                  | 1.0         | 2.0       | 8.93%   |
| dependants burden                      | 1                  | 1.0         | 1.0       | 4.46%   |
| self declared recent credit activity   | 1                  | 0.6         | 0.6       | 2.68%   |

each weight is $w_i = \text{raw}_i / 22.4$, eg for affordability, $w = 5.0 / 22.4 = 0.2232 = 22.32\%$.

---

## 3. all factors considered, and why

**core financial factors (from the brief's 7 inputs)**

- **affordability (debt to income ratio)** - strongest real signal you actually have, how much salary is already tied up in mortgage, cc spend and other loans
- **savings buffer & trend** - shows resilience to a shock, not just current comittments, a single snapshot number hides if its growing or shrinking
- **employment stability** - status, sector and tenure together, stands in for repayment reliabilty since real payment history isnt available
- **debt composition (credit mix)** - how many debt types held and revolving vs instalment split
- **dependants burden** - number and rough age of dependants, adjusts disposable income
- **housing status** - mortgage owner vs renter, same outgoings can mean very different risk underneath

**self declared factors (standing in for stuff only a real bureau would have)**

- **self declared payment history** - unverified but still the closest thing to real payment history, the biggest real world FICO factor, so leaving it out entirely felt wrong
- **self declared credit history length** - stands in for account age, which the brief gives you no way to get otherwise
- **self declared recent credit activity** - stands in for new credit/inquiries, lowest weight of all of them since its both self reported and the weakest signal here

**left out entirely (no real proxy possible)**

- **interest rates, regional cost of living, income variability, macro stuff** - all real factors in actual credit risk but out of scope for a fixed simple input set, noted as a limitation rather than forced into the formula