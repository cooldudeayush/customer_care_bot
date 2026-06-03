# Demo Script

> The one scenario that shows off every differentiator. Script it, rehearse it, record it.
> (Phase 10 finalizes this.)

## Scenario: "The angry double-charge customer who also needs an address change."

1. **Customer opens angry (Hinglish + multi-intent + anger):**
   *"I was charged TWICE for order 1234 and I'm furious. Also I need my delivery address
   changed, kuch toh karo."*
   → Emotion layer detects anger → calm, apologetic tone. Multi-intent detected.

2. **Bot traverses the graph:** order 1234 → payments → finds the duplicate charge.
   → Grounded in real (mock) data.

3. **Bot proposes the fix and CONFIRMS:** *"I can see the duplicate charge of ₹1,499 and
   I can refund it now — shall I go ahead?"* → yes → `issue_refund` runs → graph flips.

4. **Bot handles the second intent:** updates the delivery address.

5. **Bot recalls memory:** *"I also see your earlier order was delayed last month — that
   one's been delivered now, just confirming it reached you."*

6. **Out-of-scope request → graceful escalation** with a handoff packet.
   *"I've summarized everything for a specialist — you won't need to repeat any of this."*

**End state:** angry customer leaves calm, double-charge refunded, address fixed, nothing
repeated, clean human handoff. The whole pitch in ~90 seconds.

---

### Recording checklist
- [ ] Seed data loaded (customer + order 1234 + duplicate payment + a past delayed order).
- [ ] Run live once end-to-end.
- [ ] Record screen + narration.
- [ ] Put video link + this script in the README.
