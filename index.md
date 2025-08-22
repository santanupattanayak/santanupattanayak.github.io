---
layout: default
---

<div class="content-container">
  <h1>Welcome to "Symmetries & Signals"</h1>

  Welcome to my Blog space "Symmetries & Signals" where I unpack techniques and concepts in machine learning, quantum computing, optimization, and genAI with a mathematical touch to them. Whether you're here for a deep dive into diffusion models or a quick intuition on vanishing gradients, I hope you find something that sparks curiosity.

  ---

  ## Recent Posts

  {% for post in site.posts limit:5 %}
  -  {{ post.date | date: "%Y-%m-%d" }} — [{{ post.title }}]({{ post.url | relative_url }})
  {% endfor %}

  ---

  ## Topics I Explore

  - Machine Learning & Deep Learning 
  - Quantum Machine Learning 
  - Optimization & Domain Adaptation  
  - Diffusion Models & Spatial Reasoning 
  - LLMs 
  - Philosophy of Intelligence

  ---

  ## Books Authored by Me

  - [Pro Deep Learning with Tensorflow](https://link.springer.com/book/10.1007/978-1-4842-8931-0)
  - [Quantum Machine Learning](https://link.springer.com/book/10.1007/978-1-4842-6522-2)

  ---

  ## Get in Touch

  Have feedback, questions, or just want to say hi?  
  Reach out via [GitHub](https://github.com/santanupattanayak1) or drop me a note on [LinkedIn](https://www.linkedin.com/in/santanupattanayak/).

  ---

  *Knowledge is not memorized – it is realized. Saraswati sits with the still-hearted, not the loud-minded.*

</div> <!-- Closing the content-container div -->

