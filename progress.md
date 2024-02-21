yy.mm.dd
# fucking 7-th day
> 24.02.09 18:30
at fucking "Logical device and queues"

all bullshit in this time is basically 

        initWindow();
        initVulkan();
        mainLoop();
        cleanup();

initWindow is just GLFW bullshitery, don't care

initVulkan tho is

    void initVulkan() { 
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
    }

createInstance is what it says. Creation of instance.
The real HARD part is before that. By before that at this moment i mean:
 - Validation layers. As far as i understood, basically just erros, warnings and etc for vulkan;
 - Custom message callback. Basically our own rerouting of stuff catched by validation errors. Allows to change what of messages you want to see, where to output and other stuff. Basically we implemented default one, but just explicitly;
 - Physical Devices. We basically just check if whatever gpu(if we have one) is compatable. Also finding queue families. What is queue families is tho, very non understandable

 ## Queue families
 First, need to know basic, is that all stuff that Vulkan does, supposed to be put in a queue of some sort. And this queues are all capable of different things.
 By different things i mean, computing, graphics, transfering etc type of commands. Each device(or gpu in that case at least) have their queues and amount of these queues.
 Each queue can do different task, sometimes multiple tasks.
 So basically we can have 2 queues to manage graphical, transfer, compute and 1 queue just for transfering.
 We find this queues(or their indicies in property in this case) and can now assign operations to them.

 Usually its not needed to use more than 1 universal queue that can handle everything, but it is possible to use as much as you can either way. Because even if you don't have enough queues on your gpu, vulkan will just put them in the existing one.

 ig that's it for today. Next is "creating logical device".
 Honestly vulkan stuff is super draining, have to learn lots of shit, that i have no idea how to work with. And i probably would'nt be able to setup this shit on my own. I don't even know if i need to know how to setup such stuff. it feels so specific, that its possible to find templates probably.
 >21:58 

 # 8th day
 >24.02.10
 >around 22:00

 didn't do much
 created logical device
 Logical device is created from physical one. Basically an interface for this one. We get like queues from physical and then(of cource using a lot of fucking create info) create logical one, from physical.
 And also create like a queue object, to which we basically assign queue family, and index(0 in this context, because we're gonna use only one of it)

 Not that hard, but i certainly wouldn't be able to do it myself rn. Ig its normal, cause i read a relatable comment on this site, so yeah.
 Also found this from comment https://github.com/David-DiGioia/vulkan-diagrams. Might checkout later.
 >end at 23:13

 # Day 9
 > started at 15:08

i have no idea what i was doing before, so yah.
**Finally drawing stuff**! maybe. considering that drawing chapter is in 2 big chapters ahead. After 12 days. To be fair i didn't really do much in that days, so that's fine.

We have to show vulkan that glfw exists and show our vulkan surface on glfw window.
Not that hard, except needing to rework queue to add queues that support windows.
Also need to create swapchain and make sure swapchain is possible on that device. And for some reason swapchain is an exctension that's needed to be turn on.


>17:35 stopped

>22:52 i have no idea how this extent works Anyways thats tru end or whatever

# 10-th day
i did zero shit

# 11-th day
-1 shit

# 12-th day
found mistake in chooseSwapPresentMode

I didnt return eFifo :facepalm:

Validation layers are gift on this earth honestly

also 11:41, didnt really started working cause there will be extermination, just a bit of 20 minute something ig it will be. Day didnt really start.

# 13-th day
-empty

# 14-th day
>started at 13:02

W-coordinate basically computed, to show how far verticle is it from camera.

>14:32 didnt really do shit except creating swapChainImageView vector
>16:35 continuation
>17:01 +-
>18:23 -
>19:23 +-
>20:00 -
>20:53 -

# 15-th day
empty

# 16-th day
>11:51 24.02.18

it is dumb to do i decided, so yah, im gonna only write when something actually happens

Okay, so this something happend. Well duh, no shit, i wouldn't write if it didn't. Anyways

My program couldn't find shaders because this shit looks for files in build directory of this dumbassery. So i decided to change CMakeLists.txt so it compiles straight to build/shaders directory, so i can just use ../ in filename. uhums
>14:15 BREAK TILL 16:27 and STOP at 17:12

# 17-th day 
almost nothing

# 18-th day
> 8:00

pipeline i suppose
smth-smth renderPass

fucking hell i barely understand what's happening

> 12:55

> Worked 18:20 till 19:56

# 19-th day

11:58 - triangle is finally done. I drawed it. HOLY SHIT.

framesinflight

and now i will try to create a diagram for everything that happenend right now. duh
Unstoppable. super epic. 100% right i suppose.

why do i even record time i have no idea. but why not sometimes ig. i could use like toogl or smth for this.
==aaaargh, doesnt matter. Since now i wont record time. that is final. im just gonna use timestamps max==

