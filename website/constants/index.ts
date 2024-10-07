import { FaGithub } from "react-icons/fa";

export const SKILL_DATA = [
  {
      skill_name: "Music",
      image: "music.png",
      width: 1000,
      height: 1000,
    },
] as const;

export const FOOTER_DATA = [
  {
    title: "SpaceApp 2024",
    data: [
            {
              // name: "",
              // icon: FaYoutube,
              link: "https://youtube.com",
            },
          ]
  }
]

export const NAV_LINKS = [
  {
    title: "About Us",
    link: "#about-me",
  },
  {
    title: "Skills",
    link: "#skills",
  },
  {
    title: "Music Conversion",
    link: "#sky-app",
  },
] as const;


export const SOCIALS = [
  {
    name: "Source Code",
    link: "https://github.com/Rrocketter/spaceapp",
    icon: FaGithub,
  },
] as const;