---
layout: default
---

<div class="content-container">
  <h1>Welcome to "Symmetries & Signals"</h1>

  <p> Welcome to my Blog space "Symmetries & Signals" where I unpack techniques and concepts in machine learning, quantum computing, optimization, and genAI with a mathematical touch to them. Whether you're here for a deep dive into diffusion models or a quick intuition on vanishing gradients, I hope you find something that sparks curiosity. </p>

  ---

  <h2> Recent Posts </h2>

  {% for post in site.posts limit:5 %}
  -  {{ post.date | date: "%Y-%m-%d" }} — [{{ post.title }}]({{ post.url | relative_url }})
  {% endfor %}

  ---

  <h2> Topics I Explore </h2>

  <p> - Machine Learning & Deep Learning  </p>
  <p> - Quantum Machine Learning </p>
  <p> - Optimization & Domain Adaptation  </p>
  <p> - Diffusion Models & Spatial Reasoning </p>
  <p> - LLMs </p>
  <p> - Philosophy of Intelligence </p>

  ---

  <h2> Books Authored by Me </h2>

  <p> - [Pro Deep Learning with Tensorflow](https://link.springer.com/book/10.1007/978-1-4842-8931-0) </p>
  <p> - [Quantum Machine Learning](https://link.springer.com/book/10.1007/978-1-4842-6522-2) </p>

  ---

  <h2> Get in Touch </h2>

  <p> Have feedback, questions, or just want to say hi?  </p>
  <p> Reach out via [GitHub](https://github.com/santanupattanayak1) or drop me a note on [LinkedIn](https://www.linkedin.com/in/santanupattanayak/). </p>

  ---

  <p> *Knowledge is not memorized – it is realized. Saraswati sits with the still-hearted, not the loud-minded.* </p>

</div> <!-- Closing the content-container div -->

