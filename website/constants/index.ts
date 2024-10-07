import { FaGithub } from "react-icons/fa";

export const SKILL_DATA = [
  {
      skill_name: "Music",
      image: "music.png",
      width: 1000,
      height: 1000,
    },
] as const;


export const NAV_LINKS = [
  {
    title: "About Us",
    link: "#hero",
  },
  {
    title: "Skills",
    link: "#skills",
  },
  {
    title: "Music Conversion",
    link: "#sky",
  },
] as const;


export const SOCIALS = [
  {
    name: "Source Code",
    link: "https://github.com/Rrocketter/spaceapp",
    icon: FaGithub,
  },
] as const;