const cursor1 = document.querySelector('#cursor1');
const cursor2 = document.querySelector('#cursor2');

document.addEventListener('mousemove', e => {
    cursor1.setAttribute("style", "top: "+(e.pageY - 2.5)+"px; left:"+(e.pageX - 2.5)+"px;")
    cursor2.setAttribute("style", "top: "+(e.pageY - 48)+"px; left:"+(e.pageX - 48)+"px;")
})

document.addEventListener('click', () => {
    cursor2.classList.add("animate-cursorAnimation");
    setTimeout (() => {
        cursor2.classList.remove("animate-cursorAnimation");
    }, 500)
})

const navLinkEls = document.querySelectorAll('.nav_link');  
const sectionEls = document.querySelectorAll('.section');

let currentSection = 'home';
window.addEventListener('scroll', () => {
    sectionEls.forEach(sectionEl => {
        if (window.scrollY >= (sectionEl.offsetTop - sectionEl.clientHeight / 2)) {
            currentSection = sectionEl.id;
        }
    });

    navLinkEls.forEach(navLinkEl => {
        if (navLinkEl.href.includes(currentSection)) {
            navLinkEls.forEach(link => link.classList.remove('active'));
            navLinkEl.classList.add('active');  
        }
    });
});
