import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import random
import time
xml = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="agent" pos="0 0 0">
      <geom type="sphere" size="0.1" rgba ="0.2 0.2 0.9 1"/>
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="1"/>
    </body>
    <body name="target" pos="1 1 0">
      <geom type="sphere" size="0.1" rgba="0.5 0.1 0.3 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide_x" gear="1000" ctrlrange="-1 1"/>
    <motor joint="slide_y" gear="1000" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"
class enviremnt:
    def __init__(self):
      self.model = mujoco.MjModel.from_xml_string(xml)
      self.data  = mujoco.MjData(self.model)
      self.target_loc = self.data.body("target").xpos

    def reset(self):
        self.data.qpos[0]=np.random.uniform(-3,3)
        self.data.qpos[1]=np.random.uniform(-3,3)
        self.data.qvel[0]=0.0
        self.data.qvel[1]=0.0
        self.model.body("target").pos[0]=np.random.uniform(-3,3)
        self.model.body("target").pos[1]=np.random.uniform(-3,3)
        mujoco.mj_forward(self.model, self.data)
        return self.get_object()
    def get_object(self):
        return np.concatenate([
        self.data.qpos[:2],
        self.data.qvel[:2],
        self.target_loc
    ])

    def step(self,action):
        self.data.ctrl[0]=action[0]
        self.data.ctrl[1]=action[1]
        mujoco.mj_step(self.model, self.data)
        distance=np.sqrt((self.target_loc[0]-self.data.qpos[0])**2+(self.target_loc[1]-self.data.qpos[1])**2)
        reword=-distance
        done=True if distance<0.2 else False
        return self.get_object(),reword,done

class buffer :
    def __init__(self,capacisty):
        self.buffer=deque(maxlen=capacisty)
    def addd(self,current_state,action,next_state,done,reword):
        self.buffer.append((current_state,action,next_state,done,reword))
    def randoom(self,batch):
        return random.sample(self.buffer,batch)
    def __len__(self):
        return len(self.buffer)
class critic1(nn.Module):
    def __init__(self ):
        super().__init__()
        self.input=nn.Linear(9,32)
        self.layer1=nn.Linear(32,64)
        self.layer2=nn.Linear(64,64)
        self.output=nn.Linear(64,1)
    def forward(self,state,action):
        x=torch.cat([state,action],dim=-1)
        x=torch.relu(self.input(x))
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=self.output(x)
        return x
class critic2(nn.Module):
    def __init__(self ):
        super().__init__()
        self.input=nn.Linear(9,32)
        self.layer1=nn.Linear(32,64)
        self.layer2=nn.Linear(64,64)
        self.output=nn.Linear(64,1)
    def forward(self,state,action):
        x=torch.cat([state,action],dim=-1)
        x=torch.relu(self.input(x))
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=self.output(x)
        return x
class actor(nn.Module):
    def __init__(self ):
        super().__init__()
        self.input=nn.Linear(7,32)
        self.layer1=nn.Linear(32,64)
        self.layer2=nn.Linear(64,64)
        self.output=nn.Linear(64,4)
    def forward(self,x):
        x=torch.relu(self.input(x))
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=self.output(x)
        std=x[...,:2]
        mean=x[...,2:]
        std=torch.clamp(std,-20,2)
        std=torch.exp(std)
        return std , mean   
gamma=0.99
alpha=0.2
capcity=10000
buf=buffer(capcity)
batch=64
act=actor()
cri1=critic1()
tar1=critic1()
tar1.load_state_dict(cri1.state_dict())
cri2=critic2()
tar2=critic2()
tar2.load_state_dict(cri2.state_dict())
optimzer_cri1=torch.optim.Adam(cri1.parameters(),lr=3e-4)
optimzer_cri2=torch.optim.Adam(cri2.parameters(),lr=3e-4)
optimzer_act=torch.optim.Adam(act.parameters(),lr=3e-4)
tau=0.005
loss=nn.MSELoss()
env=enviremnt()
reward_history=[]
train=False
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    if train==True:
        for episode in range(10000):
            done=False
            steps=0
            total_reword=0
            current_state=env.reset()
            while done == False and steps < 500:
                steps+=1
                std,mean=act(torch.FloatTensor(current_state))
                dist=torch.distributions.Normal(mean,std)
                raw_action=dist.sample()
                action=torch.tanh(raw_action)
                next_state,reword,done=env.step(action.numpy())
                viewer.sync()
                buf.addd(current_state,action,next_state,done,reword)
                current_state=next_state
                if len(buf)> batch:
                    buff=buf.randoom(batch)
                    current_states,actions,next_states,dones,rewords=zip(*buff)
                    current_states=torch.FloatTensor(current_states)
                    actions=torch.stack(actions)
                    next_states=torch.FloatTensor(next_states)
                    dones=torch.FloatTensor(dones)
                    rewords = torch.FloatTensor(rewords)
                    with torch.no_grad():
                        std,mean=act(next_states)
                        dist=torch.distributions.Normal(mean,std)
                        raw_action=dist.sample()
                        action=torch.tanh(raw_action)
                        per_joint_log_prob = dist.log_prob(raw_action) - torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6)
                        log_prob=per_joint_log_prob.sum(dim=-1)
                        target=rewords+gamma*(torch.min(tar1(next_states,action).squeeze(),tar2(next_states,action).squeeze())-alpha*log_prob)*(1-dones)
                    optimzer_cri1.zero_grad()
                    losss1=loss(cri1(current_states,actions).squeeze(),target)
                    losss1.backward()
                    optimzer_cri1.step()
                    optimzer_cri2.zero_grad()
                    losss2=loss(cri2(current_states,actions).squeeze(),target)
                    losss2.backward()
                    optimzer_cri2.step()
                    for tw , cw in zip(tar1.parameters(),cri1.parameters()):
                        tw.data.copy_(cw.data*tau+tw*(1-tau))
                    for tw , cw in zip(tar2.parameters(),cri2.parameters()):
                        tw.data.copy_(cw.data*tau+tw*(1-tau))
                    std,mean=act(current_states)
                    dist=torch.distributions.Normal(mean,std)
                    raw_action=dist.rsample()
                    action=torch.tanh(raw_action)
                    per_joint_log_prob = dist.log_prob(raw_action) - torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6)
                    log_prob=per_joint_log_prob.sum(dim=-1)
                    optimzer_act.zero_grad()
                    losss3=(log_prob*alpha-torch.min(cri1(current_states,action).squeeze(),cri2(current_states,action).squeeze())).mean()
                    losss3.backward()
                    optimzer_act.step()
                total_reword+=reword
            reward_history.append(total_reword)
            print(f"episode = {episode} || steps = {steps} || std = {std.mean().item()} || total_reward = {total_reword.mean()}")
            if episode % 100 == 0:
                print(f"AVG last 100 = {sum(reward_history[-100:])/100:.2f}")
            if episode % 50 ==0:
                torch.save(act.state_dict(), 'mujjacttMUJOCO.pth')
                torch.save(cri1.state_dict(), 'mujjcri1MUJUCO.pth')
                torch.save(cri2.state_dict(), 'mujjcri2MUJUCO.pth')
    else:
        act.load_state_dict(torch.load("mujjacttMUJOCO.pth"))
        act.eval()
        for episode in range(10):
            done=False
            steps=0
            total_reward=0
            current_state=env.reset()
            while done == False and steps < 500:
                with torch.no_grad():
                    steps+=1
                    std,mean=act(torch.FloatTensor(current_state))
                    dist=torch.distributions.Normal(mean,std)
                    raw_action=dist.sample()
                    action=torch.tanh(raw_action)
                next_state,reword,done=env.step(action.numpy())
                viewer.sync()
                time.sleep(0.1)
                current_state=next_state
                total_reward += reword
            print(f"episode {episode} → reward = {total_reward:.2f},  → steps = {steps}")