/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package runner

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tracecapture"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

func TestRunner_TraceCaptureProducerRegistrationAndStability(t *testing.T) {
	outDir := t.TempDir()
	configYAML := fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: token-producer
    parameters:
      backend: vllm
      modelName: Qwen/Qwen2.5-0.5B-Instruct
  - type: %s
    parameters:
      outputDir: %q
      blockSizeTokens: 64
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: max-score-picker
`, tracecapture.PluginType, outDir)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = configYAML
	opts.PoolName = "trace-capture-test-pool"

	r := NewRunner()
	rawCfg, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawCfg, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err)

	plg := r.PluginHandle.Plugin(tracecapture.PluginType)
	require.NotNil(t, plg)
	t.Cleanup(func() {
		if closer, ok := plg.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
	})

	// Alpha stability requires AllowExperimentalPlugins=true.
	err = fwkplugin.ValidatePluginStability(r.PluginHandle, false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), tracecapture.PluginType)

	require.NoError(t, fwkplugin.ValidatePluginStability(r.PluginHandle, true))
}
