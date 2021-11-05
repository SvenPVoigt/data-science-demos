import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

months = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 

fname = 'co2_mm_mlo.csv'

custcol = "#ffa82d"
moncol = 'tab:cyan'
linecol = "k"

data = pd.read_csv(fname, index_col=False)
print(data.head())

print(data.month-1, months[data.month-1])


print(data.year, data.average)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=data, x=data.decdate, y=data.average, color=moncol)
sns.lineplot(data=data, x=data.decdate, y=data.interpolated, color=linecol, linestyle='-')
# plt.plot(all_years, min_vals, c ='blue', label="Minimum monthly anomaly")
# plt.plot(all_years, max_vals, c ='red', label="Maximum monthly anomaly")
plt.legend(labels=["Monthly Average", "Rolling average"], loc='lower right')
plt.xlabel('Year')
plt.ylabel('$CO_2$ ppm', horizontalalignment='right', rotation=0)
plt.title('Atmospheric $CO_2$ at Mauna Loa Observatory')

axins = ax.inset_axes([0.1, 0.6, 0.3, 0.3])
axins.plot(data.decdate, data.average, '.-', color=moncol)
axins.plot(data.decdate, data.interpolated, '.-', color=linecol)

axins.tick_params(axis='both', which='major', labelsize=10)
# axins.set_xticklabels(months[data.month - 1])
axins.set_xlim(1998, 2002) # Limit the region for zoom
axins.set_ylim(363, 375)
axins.set_title("Seasonal variation")

ax.indicate_inset_zoom(axins, edgecolor="black")


# axins.axis(visible=False)  # Not present ticks
# axins.axis(visible=False)


plt.tight_layout()

plt.savefig('climate_plot.png', dpi=300)


plt.show()


