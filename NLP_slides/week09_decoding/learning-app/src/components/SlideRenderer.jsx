/**
 * Simple Slide Renderer using MUI components
 * Displays slide content from extracted JSON
 */

import { Box, Typography, Card, CardContent, List, ListItem, ListItemText } from '@mui/material';
import { Document, Page, pdfjs } from 'react-pdf';
import { BlockMath } from 'react-katex';
import { motion } from 'framer-motion';
import parseMarkdown from '../utils/parseMarkdown.jsx';
import 'katex/dist/katex.min.css';

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const SlideRenderer = ({ slide }) => {
  if (!slide) return null;

  const figures = slide.sections?.filter(s => s.type === 'figure') || [];
  const lists = slide.sections?.filter(s => s.type === 'list') || [];
  const formulas = slide.sections?.filter(s => s.type === 'formula') || [];
  const textSections = slide.sections?.filter(s => s.type === 'text' || s.type === 'formattedText') || [];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card elevation={0} sx={{ minHeight: '70vh', bgcolor: 'background.paper' }}>
        <CardContent sx={{ p: 4 }}>
          {/* Title */}
          {slide.title && (
            <Typography variant="h4" gutterBottom color="primary" fontWeight="bold">
              {slide.title}
            </Typography>
          )}

          {/* Figures */}
          {figures.map((fig, i) => (
            <Box key={i} sx={{ my: 3, display: 'flex', justifyContent: 'center' }}>
              <Document file={`/figures/${fig.path}`}>
                <Page pageNumber={1} width={Math.min(fig.width * 900, 800)} />
              </Document>
            </Box>
          ))}

          {/* Formulas */}
          {formulas.map((formula, i) => (
            <Box key={i} sx={{ my: 3, p: 3, bgcolor: 'grey.100', borderRadius: 2, borderLeft: 4, borderColor: 'primary.main' }}>
              <BlockMath math={formula.content} />
            </Box>
          ))}

          {/* Text */}
          {textSections.map((text, i) => (
            <Typography key={i} variant="body1" sx={{ my: 2 }} color="text.primary">
              {parseMarkdown(text.content)}
            </Typography>
          ))}

          {/* Lists */}
          {lists.map((list, i) => (
            <List key={i} sx={{ my: 2 }}>
              {list.items.map((item, j) => (
                <ListItem key={j}>
                  <ListItemText
                    primary={parseMarkdown(item)}
                    primaryTypographyProps={{ variant: 'body1' }}
                  />
                </ListItem>
              ))}
            </List>
          ))}

          {/* Bottom Note */}
          {slide.bottomNote && (
            <Box sx={{ mt: 4, pt: 2, borderTop: '2px solid', borderColor: 'secondary.light' }}>
              <Typography variant="caption" fontStyle="italic" color="text.secondary">
                {slide.bottomNote}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default SlideRenderer;
