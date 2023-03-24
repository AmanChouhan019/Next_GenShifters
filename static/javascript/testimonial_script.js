const testimonialsContainer = document.querySelector(".testimonials-container");
const testimonial = document.querySelector(".testimonial");
const userImage = document.querySelector(".user-image");
const username = document.querySelector(".username");
const role = document.querySelector(".role");

const testimonials = [
  {
    name: "Miyah Myles",
    position: "Relocated from Chennai to Bengaluru.",
    photo:
      "https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-0.3.5&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=200&fit=max&s=707b9c33066bf8808c934c8ab394dff6",
    text: "Assured us of their best services, sanitization, safety and ensured they made the entire process nothing less than a breeze.",
  },
  {
    name: "June Cha",
    position: "Relocated from Bengaluru to Pune",
    photo: "https://randomuser.me/api/portraits/women/44.jpg",
    text: "Best in class packing of goods. On-time pick-up and seamless communication from the customer support team.",
  },
  {
    name: "Iida Niskanen",
    position: "Relocated from Hyderabad to Chennai",
    photo: "https://randomuser.me/api/portraits/women/68.jpg",
    text: "The people were very friendly and supportive during the shifting process. They handled everything with care. 100% would recommend them for household shifting stuff!",
  },
  {
    name: "Renee Sims",
    position: "Relocated from Punjab to Chennai",
    photo: "https://randomuser.me/api/portraits/women/65.jpg",
    text: "Amazing service!! Shifted my home without any fuss, crew was very professional and friendly. They have done fabulous job in packing and unpacking, that too at a very cheap price.",
  },
  {
    name: "Jonathan Nunfiez",
    position: "Relocated from Mumbai to Pune",
    photo: "https://randomuser.me/api/portraits/men/43.jpg",
    text: "Everything was nicely planned and executed. I was getting time to time updates on everything by call, messages, emails.",
  },
  {
    name: "Sasha Ho",
    position: "Relocated from Hyderabaad to Kolkata",
    photo:
      "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg?h=350&auto=compress&cs=tinysrgb",
    text: "Did all the things right on time. Loading & Unloading took very less time and relocation happened with zero breakage.",
  },
  {
    name: "Veeti Seppanen",
    position: "Relocated from Rajasthan to Gujarat",
    photo: "https://randomuser.me/api/portraits/men/97.jpg",
    text: "On-time pick-up and seamless communication from the customer support team.",
  },
];

let idx = 1;

function updateTestimonial() {
  const { name, position, photo, text } = testimonials[idx];

  testimonial.innerHTML = text;
  userImage.src = photo;
  username.innerHTML = name;
  role.innerHTML = position;

  idx++;

  if (idx > testimonials.length - 1) {
    idx = 0;
  }
}

setInterval(updateTestimonial, 10000);
