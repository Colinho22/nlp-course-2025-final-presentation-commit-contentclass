/**
 * Math Slide - Type 5
 * Formula-focused with explanation
 */

import { Box, Typography, Paper } from '@mui/material';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { motion } from 'framer-motion';

const katexMacros = {
  "\\given": "\\mid",
  "\\argmax": "\\operatorname*{argmax}",
  "\\softmax": "\\operatorname{softmax}",
};

const MathSlide = ({ title, formula, explanation, characteristics }) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Box>
        <Typography variant="h4" color="primary" gutterBottom fontWeight="bold" sx={{ mb: 3 }}>
          {title}
        </Typography>
        <Paper elevation={2} sx={{ p: 4, bgcolor: 'grey.50', borderLeft: 6, borderColor: 'primary.main' }}>
          <BlockMath math={formula} macros={katexMacros} />
        </Paper>
        {explanation && (
          <Typography variant="body1" sx={{ mt: 3, fontSize: '1.1rem' }}>
            {explanation}
          </Typography>
        )}
        {characteristics && (
          <Box sx={{ mt: 3 }}>
            {characteristics.map((char, i) => (
              <Typography key={i} variant="body2" sx={{ mb: 1 }}>
                • {char}
              </Typography>
            ))}
          </Box>
        )}
      </Box>
    </motion.div>
  );
};

export default MathSlide;
