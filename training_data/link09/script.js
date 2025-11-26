
$(function () {
  tabBtn();

  //notice tab
  $(document).on('click', 'div[class*="link"] .titTab li > a', function (e) {
    var contents = $(this.hash);

    $(this).addClass('current').parent('li').siblings().find('a').removeClass('current');
    $(contents).addClass('on').siblings().removeClass('on');
    e.preventDefault();
  });
});

function tabBtn() {

  $(document).on('click', '.titTab li > a:not(.btn_more)', function (e) {

    let contents = $(this.hash);

    $(this).append("<em class='hid'> 선택된 탭</em>").parent('li').addClass('current').siblings('li').removeClass('current').find('a .hid').remove();
    $(contents).addClass('on').siblings().removeClass('on');
    if ($(contents).find('.slick-slider').length) {
      $(contents).find('.slick-slider').slick('setPosition');
    }
    e.preventDefault();
  });
}
