import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Concatenate, Add, Lambda
from keras.optimizers import Adam
from keras import initializers
import keras.backend as K
import tensorflow as tf
import random
from collections import deque
from utils import Memory
from tqdm import trange

ENV_NAME='Pendulum-v0'

def ou_noise(x, mu=0, theta=0.6, sigma=0.3):
    return theta * (mu - x) + sigma * np.random.randn(1)


class Actor:
    def __init__(self, env, sess, hidden_units=32,  
                 tau=0.01, learning_rate=0.001):
        self.env=env
        self.sess=sess
        K.set_session(sess)
        self.tau=tau       
        self.learning_rate=learning_rate
        
        self.model, self.states=self.create_actor(hidden_units, True)
        self.target_model, _ = self.create_actor(hidden_units, False)
        
        self.action_grad=tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])
        params_grad=tf.gradients(self.model.output, self.model.trainable_weights, -self.action_grad)
        self.optimize=tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(params_grad, self.model.trainable_weights))
        
    def create_actor(self, hidden_units, trainable=True):
        s=Input(self.env.observation_space.shape)
        h1=Dense(hidden_units, activation='relu',
                 #kernel_initializer=initializers.truncated_normal(stddev=0.3),
                 #bias_initializer=initializers.constant(0.01), #trainable=trainable
               )(s)
        h2=Dense(hidden_units, activation='relu',
                 #kernel_initializer=initializers.truncated_normal(stddev=0.3),
                 #bias_initializer=initializers.constant(0.01), #trainable=trainable
               )(h1)
        a=Dense(self.env.action_space.shape[0], activation='tanh',
               #kernel_initializer=initializers.random_normal(stddev=0.3),
                #bias_initializer=initializers.constant(0.01), #trainable=trainable
               )(h2)
        a0=Lambda(lambda x:x*2)(a)
        model=Model(s, a0)
        #model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model, s
    
    def update_weights(self,):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
        
    def train(self, s, grads):
        #self.explore*=self.explore_decay
        self.sess.run(self.optimize, feed_dict={self.states:s,self.action_grad:grads})
        #self.sess.run(self.optimize, feed_dict={self.states:s, a, r, s_})
        
    def get_action(self, s, test_flag=False):
        #a=self.model.predict(s)[0]*(self.env.action_space.high-self.env.action_space.low)+self.env.action_space.low
        a=self.model.predict(s)[0]
        #if test_flag:
        return a
        #return np.clip(np.random.normal(a, self.explore),-2,2)

class Critic:
    def __init__(self, env, sess, hidden_units=32, tau=0.1, learning_rate=0.01):
        self.env=env
        self.sess=sess
        K.set_session(sess)
        self.tau=tau
        self.learning_rate=learning_rate
        
        self.model, self.states, self.actions = self.create_critic(hidden_units, True)
        self.target_model, _, _ = self.create_critic(hidden_units, False) 
        self.qa_grads = tf.gradients(self.model.output, self.actions)[0]
        
    def create_critic(self, hidden_units, trainable=True):
        s=Input(self.env.observation_space.shape)
        h1=Dense(hidden_units, activation='relu',
                 #kernel_initializer=initializers.truncated_normal(stddev=0.3), 
                 #bias_initializer=initializers.constant(0), #trainable=trainable
                )(s)
        h2=Dense(hidden_units, activation='linear',
                 #kernel_initializer=initializers.truncated_normal(stddev=0.3), 
                 #bias_initializer=initializers.constant(0), #trainable=trainable
                )(h1)
        a=Input(self.env.action_space.shape)
        h3=Dense(hidden_units, activation='linear',
                 #kernel_initializer=initializers.truncated_normal(stddev=0.1), 
                 #use_bias=False, #trainable=trainable
                )(a)
        h4=Add()([h2, h3])
        h5=Dense(hidden_units, activation='relu',)(h4)#这一层居然这么重要？
        q=Dense(1,activation='linear',
                #kernel_initializer=initializers.uniform(-0.01,0.01),
               #bias_initializer=initializers.constant(0), #trainable=trainable
               )(h5)
        model=Model([s, a], q)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model, s, a
    
    def get_grads(self, s, a):
        return self.sess.run(self.qa_grads, feed_dict={self.states:s, self.actions:a})
    
    def update_weights(self,):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)


class DDPG:
    def __init__(self, env, batch_size=32, gamma=0.99, 
                 hidden_units=32, maxlen=10000, 
                 tau=0.1, actor_lr=0.001, critic_lr=0.001):
        
        self.env=env
        self.batch_size=batch_size
        self.gamma=gamma
        self.maxlen=maxlen
        
        self.sess=tf.Session()
           
        
        self.actor=Actor(env, self.sess, hidden_units, tau, actor_lr)
        self.critic=Critic(env, self.sess, hidden_units, tau, critic_lr)
        self.memory=Memory(maxlen)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.step=0
        
    def store(self, exp):
        self.memory.add(exp)
        
    def update(self, ):
        if len(self.memory.buffer)<1000:#self.batch_size:
            return
        
        self.step+=1
        
        data = self.memory.sample(self.batch_size)
        s=np.array([d[0] for d in data])
        a=np.array([d[1] for d in data])
        r=np.array([d[2] for d in data])
        s_=np.array([d[3] for d in data])
        
        a_=self.actor.target_model.predict(s_)
        target_q=self.critic.target_model.predict([s_, a_])
        #y=np.array([d[2] for d in data])
        #for i in range(self.batch_size):
        #    y[i]+=self.gamma*target_q[i]
        y=r[:,np.newaxis]+self.gamma*target_q   
        self.critic.model.train_on_batch([s, a], y)
        
        action=self.actor.model.predict(s)     
        grads=self.critic.get_grads(s, action)
        self.actor.train(s,grads)
        
        if self.step%10==0:
            self.actor.update_weights()
            self.critic.update_weights()
        
        
    def get_action(self, s):
        return self.actor.get_action(s)


env=gym.make(ENV_NAME)#.unwrapped
ddpg=DDPG(env,maxlen=2000)




if __name__=='__main__':
    explore=1
    step=0
    for episodes in range(1000):
        total=0
        s=env.reset()
        for step in range(200):
            if explore>0:
                explore-=1/2000
                #explore*=0.9995
            a=ddpg.get_action(s[np.newaxis,:])
            #if episodes%10==0:
            #    env.render()
            #else:
            #a=np.clip(np.random.normal(a, explore),-2,2)
            #a=np.clip(a+(-1**np.random.randint(2))*explore,-2,2)
            a=np.clip(a+ou_noise(a)*max(0,explore),-2,2)
            s_, r, done, _ = env.step(a)
            ddpg.store([s, a, r/10, s_])
            ddpg.update()
            s=s_
            total+=r
        print('%.2f'%explore,episodes,total)
        #ddpg.actor.target_model.set_weights(ddpg.actor.model.get_weights())
        #ddpg.critic.target_model.set_weights(ddpg.critic.model.get_weights())