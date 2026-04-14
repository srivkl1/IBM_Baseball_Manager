# Baseball data here and so called API's from Statcast, fangraphs, MLB blahblahblah

# Verifying panda exists
import pandas as pd
print(pd.__version__)


from pybaseball import statcast
from pybaseball import playerid_lookup
from pybaseball import  statcast_pitcher

from pybaseball import schedule_and_record

df = playerid_lookup('schwarber', 'kyle')
print(df)

# Data for schedule for 2026 Dodgers
df_2 = schedule_and_record(2026, 'LAD')  # team = Dodgers
print(df_2)


