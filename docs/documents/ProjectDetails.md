Job Details

\[Summary\]  
We are seeking a senior-level Animation Model Specialist to engage in a focused 6-week project. The specialist will validate the feasibility of fine-tuning an open-source video diffusion model to enhance our in-house animation engine for stylized 2D character animations.

\[General Information\]  
Our company is a character IP brand with a robust short-form animation pipeline, boasting hundreds of videos produced over the past three years. We aim to explore the potential of customizing video models to support our animation processes. The project is in the initial validation phase, and we are seeking talent with expertise in 2D stylistic animation, particularly anime or cartoon styles. The role is flexible regarding location, provided there's sufficient overlap for communication.

\[Tasks and Deliverables\]  
\- Recommend a base model for stylized 2D character animation, providing rationale.  
\- Curate and caption a training set from our existing library and character T-sheets.  
\- Train one style LoRA and one character LoRA.  
\- Deliver a test set of short-form video and looping GIF outputs.  
\- Document findings and recommend a path for Phase 1\.  
\- Ensure continuous communication and progress tracking using tools like Slack.

\[Required Experience\]  
\- Proven experience in training video LoRAs, particularly for stylized 2D character animation.  
\- Strong understanding of various animation models and frameworks.  
\- Demonstrated ability to produce a reel of stylized 2D character work.  
\- Comfort with diffusion-pipe or equivalent training tooling.  
\- Ability to write clear and concise technical reports.  
\- Experience working in a fast-paced, fixed sprint environment.  
\- Clear communication skills and availability for collaboration during overlap hours.

\[Nice to Have\]  
\- Experience with ToonBoom software.  
\- Past experience with animation engine development.  
\- Familiarity with IP transfer processes.

Hey Saksham, 

Answers below in chronological order. 

Which models (both open-source and closed-source) have you experimented with? A brief note on how you tested each, along with the key pros, cons, or blockers you encountered, would be very helpful.  
Some folks used Kling 3.0, whatever was on Higgsfield. Best ones were Sora and Seedance but both lacked continuity and consistency. Looking at Krea right now.

What does the end-to-end technical setup currently look like? Even if it’s still evolving, I’d like to understand which workflow or configuration has produced the best results for your use case so far.  
We start off with concepting the skit. Then we move to produce animatics and send those off to sound dept to start scoring it. Then we work on art layout. We have a library of props / backdrops that make a regular appearance. Then we move to animation. We layout the shots and penguins. Pass that along to sound eng to refine. We then composite in AE and add final sound. 

Could you share at least three sets of reference images and videos? Ideally, each set would include 2–4 images along with a corresponding video. This will help me assess output quality and pinpoint issues like flickering or character drift.  
Here is our Instagram, take a look at what we are currently developing. Here is our GIPHY library. Since everything is done manually, GIPHY would be the first channel we'd try to augment. 

I’d also love to learn more about the competitor you mentioned. Any details about their platform and the specific features you think are worth benchmarking against would be great.  
Here is a company that has developed its own in-house engine. Take a look at their platform. However, we don't want something people can access; this is strictly for our in-house animation needs.

Additionally, please feel free to include any other technical insights or experiments that you or your team have already explored.  
None really. 

Let me know if there's anything else you need.

Data / Assets

What datasets/assets are available besides Instagram & GIPHY ?  
Roughly how many usable clips/hours of animation are available?  
Are assets organized by character, emotion, action, or scene?  
Do you have clean multi-angle character references?

Current Bottlenecks  
5\. Which stage currently takes the most time today?  
6\. Which part do you want AI to augment first?

Training / Infrastructure  
7\. Have you tried any internal fine-tuning or LoRA training before?  
8\. Are you looking for a fully open-source/self-hosted setup?  
9\. What matters most right now: quality, consistency, speed, cost, or control?  
10\. Do you want style LoRAs, character LoRAs, or both?  
11\. What GPU/cloud infrastructure is currently available for training?

Output Expectations  
12\. What output are you expecting from this sprint — GIFs, short shots, production-ready clips, or mostly internal validation?  
13\. What target duration should generated videos support?  
14\. Should the system fully generate animations or mainly assist animators within the workflow?

Evaluation  
15\. How will generated outputs be evaluated internally?  
16\. What would define a successful 6-week validation sprint for you?

Workflow / Collaboration  
17\. Could you share one complete skit example from concept → animatic → final output?  
18\. Who will be the main technical/creative point of contact during the sprint?

Data / Assets  
Besides Instagram & GIPHY, what datasets/assets are currently available internally?  
In terms of assets, we have everything from T-Poses to Master Control rules for our animation software. We have the ability to produce any assets needed in support of training the models.   
Roughly how many usable clips or hours of animation do you have access to?  
We probably have about 150+ animations that are approximately 30 seconds each. We can also produce more references etc.. as needed for training. No limitation here.   
Are assets currently organized/tagged by character, emotion, action, scene, etc.?  
Yes we have plenty of this, we can also tighten it up if needed for training purposes.  
Do you have clean multi-angle character references for the recurring penguins/characters?  
Yes, tons of this. Can go deeper into it if needed. These types of requests are not a bottleneck. 

Current Bottlenecks  
Which stage of the pipeline currently consumes the most time?  
The animation stage of bringing the animation together  
Which part would you ideally want AI to augment first?  
This is the phase where we actually animate with backgrounds, props, and posing references in place. 

Training / Infrastructure  
Have you experimented with any internal fine-tuning or LoRA training so far?  
No, but we can.   
Are you leaning toward a fully open-source / self-hosted setup?  
Honestly, whatever setup is preferred will help us achieve the results we're looking for. We're not committed to anything in particular, we're just looking for the most efficient, highly controlled, and consistent output.   
At this stage, what matters most: quality, consistency, speed, cost, or control?  
quality, consistency, control, and cost  
Would you want style LoRAs, character LoRAs, or both?  
I don't have a preference or the knowledge to determine this.   
What GPU/cloud infrastructure is currently available for training and inference?  
We don't have anything in place currently. 

Output Expectations  
What would you expect as the output from this sprint GIFs, short shots, production-ready clips, or primarily internal validation?  
Production ready clips. GIFs is a good starting point too.   
What target duration should generated videos support?  
30 seconds to 1-2 minute eventually. Right now the need is to produce up to 30 seconds for animations as well as GIFs.   
Should the system fully generate animations, or mainly function as an animator-assist tool within the existing workflow?  
Short term the ladder, long-term the former. 

Evaluation  
How will generated outputs be evaluated internally?  
Continuity, consistency, and aesthetics. Eventually, we will challenge things like camera work, etc.  
What would define a successful 6-week validation sprint for your team?  
A model that is tried and tested to meet our standards and produce consistent animations. We will be closely examining how to augment our GIF animation pipeline. If our engine achieves this in \~6 weeks, we are headed on the right path to augmenting our Instagram short form content. GIF is a good barometer to gauge consistency and production value. 

Workflow / Collaboration  
Would it be possible to share one complete skit example from concept → animatic → final output? That would help me understand the full production flow and identify the best insertion points for AI tooling.  
Yes, I can add you to Slack and connect you with the relevant stakeholders.   
Who would be the primary technical and creative point of contact during the sprint?  
Me at a high level. Our animation director and head of editorial for the day to day work.   
Please let me know next steps\!  
