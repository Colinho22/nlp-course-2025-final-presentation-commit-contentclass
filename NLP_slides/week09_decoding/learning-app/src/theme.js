/**
 * Material-UI Theme Configuration
 * Purple/Lavender color scheme matching Beamer template
 */

import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#3333B2',      // mlpurple
      light: '#ADADE0',     // mllavender
      dark: '#2525a0',
      contrastText: '#fff',
    },
    secondary: {
      main: '#ADADE0',      // mllavender
      light: '#CCCCEB',     // mllavender3
      dark: '#C1C1E8',      // mllavender2
      contrastText: '#3333B2',
    },
    success: {
      main: '#2CA02C',      // mlgreen
    },
    warning: {
      main: '#FF7F0E',      // mlorange
    },
    error: {
      main: '#D62728',      // mlred
    },
    info: {
      main: '#0066CC',      // mlblue
    },
    background: {
      default: '#F0F0F0',   // lightgray
      paper: '#ffffff',
    },
    text: {
      primary: '#404040',
      secondary: '#7F7F7F',  // mlgray
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 700,
      color: '#3333B2',
    },
    h4: {
      fontWeight: 600,
      color: '#3333B2',
    },
    h5: {
      fontWeight: 600,
    },
  },
  components: {
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#ffffff',
          borderRight: '2px solid #CCCCEB',
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          '&.Mui-selected': {
            backgroundColor: '#D6D6EF',
            borderLeft: '4px solid #3333B2',
            '&:hover': {
              backgroundColor: '#CCCCEB',
            },
          },
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          backgroundColor: '#CCCCEB',
          height: 8,
          borderRadius: 4,
        },
        bar: {
          backgroundColor: '#3333B2',
        },
      },
    },
  },
});

export default theme;
