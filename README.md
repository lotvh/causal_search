# Causal Search Algorithm
## Objective
This is an algorithm which uses causal interference on a dataset consisting of
common KPIs (Key Performance Indicators) in order to predict what caused
a significant change in the performance of a business.

## Libraries Used
- Numpy
- Pandas
- Matplotlib

## Example
A company which keeps records on personal information of its employees,
provided a dataset with following KPIs:
- team (green or red)
- commute distance
- experience
- education
- nationality
- age
- workdays/week

which shows a drastic decline in workdays/week for certain employees after
an intervention of HR.

Running the algorithm, the following results were obtained:
- for the green team, only considering one KPI:
![green_team_1](https://user-images.githubusercontent.com/49079733/190246750-0ba30b28-baac-492c-a39e-925972dd525a.png)

- for the green team, considering two KPIs:
![green_team_2](https://user-images.githubusercontent.com/49079733/190246881-248d47f0-c6c8-4e5f-9841-9d698ac3d5f9.png)

- for the red team, only considering one KPI:
![red_team_1](https://user-images.githubusercontent.com/49079733/190246933-58181eb7-fdda-487e-b8ba-adce1284585c.png)

- for the red team, considering two KPIs:
![red_team_2](https://user-images.githubusercontent.com/49079733/190247007-ed89c078-c2b2-4de0-8c96-b7ff1c6e28cb.png)

which shows that for both teams, the KPI which had the most impact on the change was the seniority level. The seniority level with label 'senior' had the biggest impact. This also shows that seniority level and age are correlated.

