---
layout: home
title: "Symmetries & Signals"
subtitle: "Explorations in Machine Learning, Deep Learning, Optimization, and Beyond"
description: "A blog on technical deep dives, elegant math, and the occasional philosophical tangent."
permalink: /
---

## 👋 Welcome

Welcome to my digital notebook—a space where I unpack ideas in machine learning, optimization, and the occasional philosophical rabbit hole. Whether you're here for a deep dive into diffusion models or a quick intuition on vanishing gradients, I hope you find something that sparks curiosity.

---

## 🧠 Recent Posts

{% for post in site.posts limit:5 %}
- 📅 {{ post.date | date: "%Y-%m-%d" }} — [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}

[→ View all posts](/blog)

---

## 🔍 Topics I Explore

- Machine Learning & Deep Learning  
- Optimization & Domain Adaptation  
- Diffusion Models & Spatial Reasoning  
- Technical Writing & Visualization  
- Philosophy of Intelligence

---

## 📬 Get in Touch

Have feedback, questions, or just want to say hi?  
Reach out via [GitHub](https://github.com/santanupattanayak) or drop me a note on [LinkedIn](https://www.linkedin.com/in/santanupattanayak/).

---

*Thanks for stopping by. Stay curious.*
