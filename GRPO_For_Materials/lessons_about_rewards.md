## Lessos about constructing a reward function for RL

It is being know for a long time that the reward construction build on top of an environment such that an objective is learned, is a non-trivial task. Out of the many problems that arise, reward hacking stands out the most, due to its catastrophic results, and its unexpected appearance. 
Why is so hard? Well, in my experience is because you first of all start with an objective that is hard to define it objectively: "what is a clean room?", "what is a good poem?", "What is a stable crystal?". Besides, limitations on the design are hard to predict, and only show up after training. 

A secondary problem is that sometimes rewards are "correctly" defined, but is the limitations of the model itself, on available resources, that complicate designing appropiate rewards. The reward designer ends up having to make strategic choices, should i let it train for more epochs? more repetitions? or should i just run fast iterations with different tweaks in the reward until something hits. As ill show in a bit, there are lessons that can be learned quickly, and lessons that take longer.

## Reward designing for Crystal Generation

First of all, as i explained earlier, this is an issue because models are not capable enough. Simple as that. If they were capable enough, a binary reward of what *I* think is a stable crystal should be enough. But they are not, at least at testing scale (0.5-8B parameter models). Even in the original DeepSeek paper, the rewards included a "format" term, that gives a small reward if the format is correct, instead of just neglecting answers that do not have it. 

First lesson: The more terms your reward have, the harder it is to control the training behaviour. 
Second lesson: Is better to avoid negative rewards, punishing a model for an error just avoids it, but doesnt enhance the desired one.
Third lesson: When mixing multiple terms in the reward, rubric style rewards are better than gated ones. 
Fourth lesson: When possible, continous reward improve training as there's less unuseful generations. 
Fifth lesson: This is more of a tip to newcomers, always try to think on what is the path of less resistance.


## Avoid punishing the model, better to reward good behaviours than punish bad ones

At the beginning of my project, I ran into the issue of seeing the generations of what could be considered a cif file, but because of different reasones like not being inside the right brackets, or including text with the answer itself, I decided to add a negative reward if the answer inside the brackets was not parsable by the CifParser object. I quickly found out this was a problem.

The model learned to only output <answer> tags, and nothing inside. Quickly realizing, that my reward design just created a trap in the hill-climbing problem, a local optimum.
The model went through its first bump pretty quickly: using the <answer> </answer> tags. It earned 0.1 points, but when exploring more options, when it got it wrong, it would deduct, 0.05 points. So suddenly, because my model was not competent enough yet. It learned that from the current policy, the best policy was to not even try. After all, 0.1 is higher than 0.05. And the very low probability of getting a "good" cif file, was not enough to promote exploration. Entropy quickly went downhill, and the model crashed. 

Remarkable, I would say, you really start thinking, wow, this models are like kids. Punish a kid for breaking its toys trying to see whats inside, and you may kill his curiosity about how things are made. Yeah, the more I work on this, the more I antropomorphised LLMs. 

## The more terms, the more unexpected the outcome

Some simple calculations can help with this. For every action, three different judges answer back with either 0 or 1. Their scores add up.
You end up with the possible scores being {0, 1, 2, 4}. Now, let's say one of the judges is the public, which usually gets weigther with a 50% discount. Now you possible scores are: {0, 0.5, 1, 1.5, 2, 2.5}. Still three terms, but now we have 50% more possible outcomes. Is this good or bad? It depends, in the first case, there are different policies that will get the same reward. While in the second, the number of policies that are non-trivially different with the same reward are reduced. 

|Policies| Judge 1 | Judge 2 | Judge 3 | Reward 1| Reward 2 |
|Policy 1|    ✅   |    ❌   |   ✅    |   2     |    1.5  |
|Policy 2|    ✅   |    ✅   |   ❌    |   2     |    1.5  |
|Policy 3|    ❌   |    ✅   |   ✅    |   2     |     2   |

Clearly, with our second reward, Policy 3 would be preferred. But this gets messy when the scores of each judge/term are related. 
Let's go back to our Crystal Generation task. I have four different terms: Format, Bond Length reasonableness, Forces, Composition, and Formation Energy.

In my first iteration, I didn't used formation energy. Composition was included cause I thought in advance I cant just allow for "free" generation, cause some crystals are easier than others: Less atoms, Fully metalic, better affinity. So I got ahead and condition the generation on a given composition. (NaCl, AgNO3, etc). 
At first, all good. Training improved, until it didnt. At some point, the reward kept improving, while i noticed that the Bond Length Forces improved, Composition was getting worse. Of course. My model has learned that disregarding the composition/constraing, and generating an easier Crystal was better. In the end, two rewards getting higher would outcompete one getting worse. 

Interestingly, from the family of policies that get the composition right, P(C=1), there's a lot less of policies that also get Forces and Bonds correctly P(B=1 F=1). There's a whole other ordeal here on how to analyze this! Even take advantage of this during training. But thats for another discussion. 

In the end, as you can tell, the more options you give a model, the crazier it can get! In my case, while i was testing smaller models, I increased weight importance to Composition to avoid rewarding the model for this undesired behaviour. 


## Rubrics are better than Gated rewards

My first iteration of my reward was a single function, in which if a check failed, it will return 0, if it passed all checks it would pass 1. Very quickly i went and added small increments of rewards for each step successfully accomplished. Order which i decided came from logic and my bias towards which step was easier to accomplish first. 

Why is this a problem? Well, looking through some outputs, i noticed at early stages that the model was outputing some interesting cif files, but because the Composition was slightly off, it would avoid any other checks. That is wastefull! Imagining rejecting a good crystal file with appropiate bonds, forces and energy, because it failed to get the constraint right, at a moment in training when im just learning how to write a cif file. So many wasteful examples that could help from the beginning. Also, the model loses a lot of exploration capabilites, because it learns one way of writing a cif file with the right composition, and then it moves on to our next sub-goal. 

Changing to a rubric style reward accelerate training across all sub-goals. 

## Continuous rewards foment learning more than continuous rewards

In my case, there is a somehow well defined value that defines a "good" vs "bad" crystal. Literature has gone with  >0.1 eV formation energy to 
assing a crystal as stable enough. But of course, 0.5 is better than 3.2. Limiting us to not assing a training signal in this case is wasteful again. Binary rewards, are used on tasks when is hard to grade a "half" good task, like a backflip. What is half good backflip? You either get it or you dont. But when available, continuous rewards allow the model to get gradually better along the respective sub-goal.