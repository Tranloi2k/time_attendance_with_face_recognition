let nav = document.querySelector('nav');
let dropdown = nav.querySelector('.dropdown');

let nav1 = document.querySelector('nav1');
let dropdown1 = nav.querySelector("[id='id1']");

let dropdownToggle = nav.querySelector("[data-action='dropdown-toggle']");
let navToggle = nav.querySelector("[data-action='nav-toggle']");
let dropdownToggle1 = nav.querySelector("[data-action='dropdown-toggle1']");
let navToggle1 = nav.querySelector("[data-action='nav-toggle1']");

dropdownToggle.addEventListener('click', () => {
	if (dropdown.classList.contains('show')) {
		dropdown.classList.remove('show');
	} else {
		dropdown.classList.add('show');
	}
})

dropdownToggle1.addEventListener('click', () => {
	if (dropdown1.classList.contains('show')) {
		dropdown1.classList.remove('show');
	} else {
		dropdown1.classList.add('show');
	}
})

navToggle.addEventListener('click', () => {
	if (nav.classList.contains('opened')) {
		nav.classList.remove('opened');
	} else {
		nav.classList.add('opened');
	}
})