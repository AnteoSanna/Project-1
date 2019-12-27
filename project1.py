import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('data.csv')

df = pd.DataFrame(data)

df
print(df.columns)

df.rename(columns = {'Percent of undergraduate students awarded federal  state  local  institutional or other sources of grant aid (SFA1718)':'percent_stdnt_aid','Grand total (S2018_IS  All instructional staff)':'total_staff','Graduation rate  Black  non-Hispanic (DRVGR2018)':'black_graduation_rate','Graduation rate  Hispanic (DRVGR2018)':'hispanic_graduation_rate','Percent admitted - total (DRVADM2017_RV)':'percent_admitted','Student-to-faculty ratio (EF2017D_RV)':'stdnt_fclty_ratio','Published in-state tuition 2017-18 (IC2017_AY)':'in_state_tuition','Published out-of-state tuition 2017-18 (IC2017_AY)':'out_state_tuition','Published in-district tuition 2017-18 (IC2017_AY)':'in_district_tuitiion','Percent Black Staff':'percent_black_staff','Percent Hispanic Staff':'percent_hispanic_staff'}, inplace = True)

# Here I transform the data colums so it is easier to refer to later

print('Column names post transformation')
print(df.columns)


#excluded universities that had total staff greater than 50

greaterthan50 = df[df.total_staff > 50]
greaterthan50

#drop institutions with empty coumn values

transformeddata = greaterthan50.dropna(how='any')

data2 = transformeddata

#do the regression for african american graduation rates!

results = smf.ols('black_graduation_rate ~ percent_black_staff + stdnt_fclty_ratio + percent_stdnt_aid + percent_admitted', data=transformeddata).fit()

print(results.summary())

#do the regression for hispanic graduation rates!

results = smf.ols('hispanic_graduation_rate ~ percent_hispanic_staff + stdnt_fclty_ratio + percent_stdnt_aid + percent_admitted', data=transformeddata).fit()

print(results.summary())
