
/* 퀵링크 */
var Qui_LinkPrev = '.Quick_Link .prev';
var Qui_LinkNext = '.Quick_Link .next';
$("#Qui_LinkSlide").slick({
  slider: 'li',
  dots: true,
  infinite: true,
  autoplay: false,
  slidesToShow: 6,
  speed: 500,
  prevArrow: Qui_LinkPrev,
  nextArrow: Qui_LinkNext,
  responsive: [
    {
      breakpoint: 1750,
      settings: {
        slidesToShow: 5
      }
    }, {
      breakpoint: 1440,
      settings: {
        slidesToShow: 4,
      }
    }, {
      breakpoint: 860,
      settings: {
        slidesToShow: 3
      }
    }, {
      breakpoint: 480,
      settings: {
        slidesToShow: 2
      }
    }
  ]
});
