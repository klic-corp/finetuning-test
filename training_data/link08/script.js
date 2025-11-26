
$(function () {

  siteLink();


  $('.siteWrap .siteBtn > button').on('click', function () {
    if ($(this).attr('title') == '확장') {
      $(this).attr('title', '축소');
    } else { $(this).attr('title', '확장'); }
  });
});
//link2
function siteLink() {
  $('.siteBtn button').click(function (e) {
    $(this).parent().siblings('div').children('div').stop().slideUp(300);
    $(this).siblings("div").stop().slideToggle(300);

    if ($(this).parent('.siteBtn').hasClass('on')) {
      $('.siteBtn').removeClass('on');
    } else {
      $('.siteBtn').removeClass('on');
      $(this).parent('.siteBtn').addClass('on');
    }
    e.preventDefault();
  });
}
