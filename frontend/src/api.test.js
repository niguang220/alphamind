import { describe, expect, it } from 'vitest'
import { isGuardrail } from './api'

/**
 * The distinction this guards is easy to get wrong and expensive when it is: a Compliance
 * answer that offers a human advisor also sets escalated, and rendering that as a guardrail
 * interception would claim the system refused a question it actually answered.
 */
describe('isGuardrail', () => {
  it('is true only when the orchestrator stamped a guardrail routing reason', () => {
    expect(isGuardrail({ routing_reason: 'guardrail=advice_request,intent=advice_request' })).toBe(true)
  })

  it('is false for a domain answer that escalated itself', () => {
    const complianceTurn = {
      escalated: true,
      agent_type: 'compliance',
      routing_reason: 'intent=suitability, group=compliance, primary=compliance, scores=[compliance=1.31]',
    }
    expect(isGuardrail(complianceTurn)).toBe(false)
  })

  it('does not fall back to the escalated flag', () => {
    expect(isGuardrail({ escalated: true, routing_reason: 'intent=valuation' })).toBe(false)
  })

  it('tolerates a missing or malformed turn', () => {
    for (const value of [undefined, null, {}, { routing_reason: null }, { routing_reason: 42 }]) {
      expect(isGuardrail(value)).toBe(false)
    }
  })
})
