
/* Quick_Menu1 */
var returntarget = '';

$(document).on('click', '.Quick_Menu1 .tabBtn a', function (e) {
  var contents = $(this.hash);
  returntarget = $(this);

  $(this).addClass('current').parent('li').siblings().find('a').removeClass('current');
  $(this).attr('title', '선택됨').parent('li').siblings().find('a').attr('title', '');
  $(contents).addClass('on').siblings().removeClass('on');

  if ($(contents).hasClass('on') == true) {
    $(contents).find('li:first').children('a').focus();
  }

  if ($(contents).find('.slick-slider').length) {
    $(contents).find('.slick-slider').slick('setPosition');
  }

  e.preventDefault();
});

$('.tabConCls').on('click', function (e) {
  $(this).parent().removeClass('on');

  if (returntarget !== '') {
    returntarget.focus();
    returntarget = '';
  }

  e.preventDefault();
});
