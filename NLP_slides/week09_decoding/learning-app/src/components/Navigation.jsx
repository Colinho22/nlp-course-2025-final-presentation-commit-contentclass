/**
 * Navigation Component
 * Prev/Next buttons at bottom of slides
 */

import { Box, Button, IconButton } from '@mui/material';
import { ArrowBack, ArrowForward, Home, LastPage } from '@mui/icons-material';

const Navigation = ({ onPrev, onNext, onFirst, onLast, canPrev, canNext }) => {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 3, bgcolor: 'white', borderTop: '1px solid', borderColor: 'divider' }}>
      {/* Left side */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <IconButton onClick={onFirst} disabled={!canPrev} color="primary">
          <Home />
        </IconButton>
        <Button
          variant="contained"
          startIcon={<ArrowBack />}
          onClick={onPrev}
          disabled={!canPrev}
        >
          Previous
        </Button>
      </Box>

      {/* Right side */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button
          variant="contained"
          endIcon={<ArrowForward />}
          onClick={onNext}
          disabled={!canNext}
        >
          Next
        </Button>
        <IconButton onClick={onLast} disabled={!canNext} color="primary">
          <LastPage />
        </IconButton>
      </Box>
    </Box>
  );
};

export default Navigation;
