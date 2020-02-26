#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mindwave, time
from serial import Serial

headset = mindwave.Headset('/dev/tty.MindWaveMobile-SerialPo','01710007800B')


# In[3]:


pwd


# In[ ]:


time.sleep(2)

headset.connect()
print("Connecting...")

while headset.status != 'connected':
    time.sleep(0.5)
    print(headset.status)
    if headset.status == 'standby':
        headset.connect()
        print ("Retrying connect...")
print("Connected.")

while True:
    print("Attention: %s, Meditation: %s" % (headset.attention, headset.meditation))


# In[12]:


get_ipython().magic(u'pinfo headset')


# In[ ]:




