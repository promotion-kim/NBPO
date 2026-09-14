from pathlib import Path
import re,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
text=(ROOT/'main_v6.tex').read_text();body=text.split('% BEGIN AUTO TAB1 BODY')[1].split('% END AUTO TAB1 BODY')[0];rows=[]
for line in body.splitlines():
 if ' & ' not in line or 'defined reference' in line:continue
 nums=re.findall(r'\$[^$]*?([0-9]+\.[0-9]+)[^$]*?\$',line)
 if len(nums)!=5:continue
 vals=list(map(float,nums))
 if 'sample SD' in line:rows[-1]['sd']=vals
 else:rows.append({'name':line.split(' & ')[0],'mean':vals})
labels=['NBPO','Fixed-reference Nash','BT-RM–Nash','Game-utilitarian','Global game-maxmin','DPO (uniform)','PROSPER (2 seeds)','MOPO adaptation'];colors=['#007f9d','#e18b2c','#8c7955','#8661bd','#d25563','#239c95','#ba59b6','#717171']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.3,2.65))
for row,name,c in zip(rows,labels,colors):
 for ax,(xi,yi) in zip(axs,[(3,0),(0,2)]):ax.errorbar(row['mean'][xi],row['mean'][yi],xerr=row['sd'][xi],yerr=row['sd'][yi],fmt='o',markersize=3,color=c,capsize=2,label=name,elinewidth=.7)
for ax in axs:ax.scatter([.5],[.5],marker='*',c='black',s=32,label='Base reference');ax.grid(alpha=.15)
axs[0].set(xlabel='Helpfulness win rate',ylabel='Instruction-following win rate');axs[1].set(xlabel='Instruction-following win rate',ylabel='Honesty win rate')
handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,fontsize=6.4);fig.tight_layout(rect=(0,.24,1,1));fig.savefig(ROOT/'figures/uf_tradeoffs.pdf');plt.close(fig)
cat={'NBPO exact':[.5327,.5129,.5061,.5442],'NBPO fitted':[.5494,.5369,.5198,.5402],'Utilitarian exact':[.5316,.5117,.5043,.5437],'Utilitarian fitted':[.5500,.5336,.5123,.5399]};fresh={'NBPO (N=186)':[.5455,.5460,.5221,.5781],'Utilitarian (N=188)':[.5158,.5279,.4926,.5575],'DPO (N=187)':[.5806,.5548,.5171,.6357]}
fig,axs=plt.subplots(1,2,figsize=(7.3,2.4),sharey=True);x=np.arange(4)
for i,(name,y) in enumerate(cat.items()):axs[0].plot(x,y,['o-','s-','o--','s--'][i],c=['#007f9d','#007f9d','#e18b2c','#e18b2c'][i],ms=3,lw=.9,label=name)
for name,y,c in zip(fresh,fresh.values(),['#007f9d','#e18b2c','#8661bd']):axs[1].plot(x,y,'o-',c=c,ms=3,lw=.9,label=name)
for ax in axs:ax.set_xticks(x,['IF','Truth','Honesty','Help']);ax.axhline(.5,c='.6',ls=':',lw=.8);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=6,loc='upper left');ax.set_ylim(.482,.65)
axs[0].set_title('Exact and fitted pools: common N=78',fontsize=8);axs[1].set_title('Fresh responses: method-specific sets',fontsize=8);axs[0].set_ylabel('Independent direct win rate');fig.tight_layout();fig.savefig(ROOT/'figures/target_transfer.pdf');plt.close(fig)
(ROOT/'figures/reconstruction_data.json').write_text(json.dumps({'table1':rows,'table4_categorical':cat,'table4_fresh':fresh},indent=2)+'\n')
print('Rebuilt two descriptive figures from reported table cells; no new measurements.')
